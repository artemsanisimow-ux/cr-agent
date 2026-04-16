"""
Continuous Research Agent
=========================
Monitors product signals between sprints so discovery never stops.
Based on Teresa Torres' Continuous Discovery Habits framework.

What it does:
1. Signal Collector  — accepts signals from any source (feedback, metrics, support, reviews)
2. OST Maintainer   — maintains Opportunity Solution Tree; maps signals to opportunities
3. Pattern Detector — detects trends, anomalies, and recurring themes across signals
4. Weekly Digest    — generates structured weekly report for the product trio
5. Alert Engine     — fires when signal volume or sentiment crosses thresholds

Feeds into: Discovery agent (enriched context), Planning agent (pre-mortem risks)

Setup:
    pip install langgraph langchain-anthropic langgraph-checkpoint-sqlite python-dotenv

Usage:
    python3 cr_agent.py --lang en          # add new signals + generate digest
    python3 cr_agent.py --lang en --digest # weekly digest only (no new signals)
    python3 cr_agent.py --lang ru
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

load_dotenv()
from i18n import get_language, get_language_instruction
from i18n import t as tr

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MODEL = ChatAnthropic(model="claude-opus-4-5", max_tokens=2048)
DB_PATH = "cr_signals.db"

# Alert thresholds
ALERT_VOLUME_THRESHOLD = 5    # same opportunity mentioned N+ times in 7 days → alert
ALERT_SENTIMENT_DROP   = 0.3  # sentiment drops by 30%+ → alert
SIGNAL_SOURCES = ["user_interview", "support_ticket", "app_review",
                   "nps_comment", "sales_call", "analytics", "other"]


# ─────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS signals (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            source       TEXT NOT NULL,
            raw_text     TEXT NOT NULL,
            opportunity  TEXT,
            sentiment    REAL,
            tags_json    TEXT,
            created_at   TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS opportunities (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            title        TEXT NOT NULL UNIQUE,
            outcome      TEXT NOT NULL,
            signal_count INTEGER NOT NULL DEFAULT 1,
            avg_sentiment REAL,
            status       TEXT NOT NULL DEFAULT 'open',
            created_at   TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS digests (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            week_start   TEXT NOT NULL,
            content      TEXT NOT NULL,
            signal_count INTEGER NOT NULL DEFAULT 0,
            created_at   TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type   TEXT NOT NULL,
            message      TEXT NOT NULL,
            opportunity  TEXT,
            resolved     INTEGER NOT NULL DEFAULT 0,
            created_at   TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


def save_signal(source: str, text: str, opportunity: str,
                sentiment: float, tags: list[str]) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "INSERT INTO signals (source, raw_text, opportunity, sentiment, tags_json) VALUES (?,?,?,?,?)",
        (source, text, opportunity, sentiment, json.dumps(tags))
    )
    sig_id = cur.lastrowid
    # Upsert opportunity
    conn.execute("""
        INSERT INTO opportunities (title, outcome, signal_count, avg_sentiment)
        VALUES (?, '', 1, ?)
        ON CONFLICT(title) DO UPDATE SET
            signal_count = signal_count + 1,
            avg_sentiment = (avg_sentiment * signal_count + excluded.avg_sentiment) / (signal_count + 1),
            updated_at = datetime('now')
    """, (opportunity, sentiment))
    conn.commit()
    conn.close()
    return sig_id


def save_alert(alert_type: str, message: str, opportunity: str = None) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO alerts (alert_type, message, opportunity) VALUES (?,?,?)",
        (alert_type, message, opportunity)
    )
    conn.commit()
    conn.close()


def save_digest(week_start: str, content: str, signal_count: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO digests (week_start, content, signal_count) VALUES (?,?,?)",
        (week_start, content, signal_count)
    )
    conn.commit()
    conn.close()


def get_recent_signals(days: int = 7) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT * FROM signals WHERE created_at >= ? ORDER BY created_at DESC",
        (cutoff,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_top_opportunities(limit: int = 10) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM opportunities WHERE status='open' "
        "ORDER BY signal_count DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_unresolved_alerts() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM alerts WHERE resolved=0 ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_db_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    stats = {
        "total_signals": conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0],
        "opportunities": conn.execute("SELECT COUNT(*) FROM opportunities WHERE status='open'").fetchone()[0],
        "digests": conn.execute("SELECT COUNT(*) FROM digests").fetchone()[0],
        "unresolved_alerts": conn.execute("SELECT COUNT(*) FROM alerts WHERE resolved=0").fetchone()[0],
        "signals_last_7d": conn.execute(
            "SELECT COUNT(*) FROM signals WHERE created_at >= ?",
            ((datetime.now() - timedelta(days=7)).isoformat(),)
        ).fetchone()[0],
    }
    conn.close()
    return stats


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

CLASSIFY_PROMPT = """{lang}You are a product researcher using Teresa Torres' Continuous Discovery framework.

Classify this signal and extract structured insight.

Signal text: {text}
Source: {source}

Return JSON only:
{{"opportunity": "concise opportunity title (user need or pain point, e.g. 'Difficulty tracking expenses on mobile')",
  "sentiment": float from -1.0 (very negative) to 1.0 (very positive),
  "tags": ["tag1", "tag2"],
  "insight": "one sentence — what this reveals about user needs",
  "assumption_to_test": "what assumption this signal challenges or confirms"}}"""

PATTERN_PROMPT = """{lang}Analyze these {n} signals from the past 7 days. Apply Teresa Torres' Continuous Discovery principles.

Signals summary:
{signals_summary}

Top opportunities by volume:
{opportunities}

Identify:
1. Emerging patterns (what keeps coming up)
2. Anomalies (unexpected signals that deserve investigation)
3. Assumptions being challenged
4. Highest-priority opportunities for the OST

Return JSON only:
{{"patterns": [{{"theme": str, "signal_count": int, "implication": str}}],
  "anomalies": [{{"signal": str, "why_notable": str}}],
  "challenged_assumptions": [str],
  "ost_priorities": [{{"opportunity": str, "evidence_strength": "strong|moderate|weak", "recommended_action": "explore|test|monitor|close"}}],
  "confidence": "high|medium|low"}}"""

DIGEST_PROMPT = """{lang}Write a weekly discovery digest for the product trio. Be concise and actionable.

Week: {week}
Total signals: {n_signals}
New signals this week: {new_signals}
Top opportunities: {opportunities}
Patterns found: {patterns}
Active alerts: {alerts}

Format:
- Headline (1 sentence — most important thing this week)
- Signal summary (2-3 sentences)
- Top 3 opportunities with evidence
- One assumption to test this week
- One user to talk to and why

Return JSON only:
{{"headline": str,
  "signal_summary": str,
  "top_opportunities": [{{"opportunity": str, "evidence": str, "action": str}}],
  "assumption_to_test": {{"assumption": str, "test_method": "interview|survey|experiment", "question": str}},
  "recommended_interview": {{"profile": str, "reason": str, "question": str}},
  "planning_risks": [str]}}"""


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

class CRState(TypedDict):
    # Mode
    digest_only: bool
    outcome: str          # product outcome we're researching for

    # Collected this session
    new_signals: list     # [{source, text, classified}]

    # Analysis
    patterns: dict
    digest: dict
    alerts_fired: list

    # Control
    needs_human: bool
    human_approved: bool
    session_id: str
    audit: list


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def L(en: str, ru: str) -> str:
    return en if get_language() == "en" else ru


def call_model(prompt: str) -> dict:
    resp = MODEL.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=tr("prompt_finalize")),
    ])
    raw = resp.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"error": "parse_failed"}


def log(state: CRState, step: str, summary: str) -> list:
    entry = {"ts": datetime.now().isoformat(), "step": step, "summary": summary}
    print(f"[{step.upper()}] {summary}")
    return state.get("audit", []) + [entry]


def _check_alerts(signals: list[dict]) -> list[dict]:
    """Fire alerts based on volume and sentiment thresholds."""
    alerts = []

    # Volume alert: same opportunity N+ times in last 7 days
    opp_counts = Counter(s["opportunity"] for s in signals if s.get("opportunity"))
    for opp, count in opp_counts.most_common():
        if count >= ALERT_VOLUME_THRESHOLD:
            msg = L(f"High signal volume: '{opp}' mentioned {count}x this week",
                    f"Высокий объём сигналов: '{opp}' упомянуто {count}x за неделю")
            alerts.append({"type": "volume", "message": msg, "opportunity": opp})
            save_alert("volume", msg, opp)

    # Sentiment alert: negative signals spike
    neg_signals = [s for s in signals if (s.get("sentiment") or 0) < -0.5]
    if len(neg_signals) >= 3:
        msg = L(f"Sentiment alert: {len(neg_signals)} strongly negative signals this week",
                f"Алерт тональности: {len(neg_signals)} сильно негативных сигналов за неделю")
        alerts.append({"type": "sentiment", "message": msg})
        save_alert("sentiment", msg)

    return alerts


# ─────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────

def gather_signals(state: CRState) -> CRState:
    """Collect new signals from the PM."""
    if state.get("digest_only"):
        audit = log(state, "gather_signals", L("Digest-only mode — skipping signal collection",
                                                "Режим дайджеста — пропускаем сбор сигналов"))
        return {**state, "new_signals": [], "audit": audit}

    print(f"\n{'=' * 60}")
    print(f"🔍 {L('Continuous Research Agent', 'Continuous Research агент')} | "
          f"{L('Session', 'Сессия')}: {state['session_id']}")

    stats = get_db_stats()
    print(f"\n📊 {L('Current state', 'Текущее состояние')}: "
          f"{stats['total_signals']} {L('signals', 'сигналов')} | "
          f"{stats['opportunities']} {L('open opportunities', 'открытых возможностей')} | "
          f"{stats['signals_last_7d']} {L('this week', 'за неделю')}")

    outcome = state.get("outcome") or input(
        f"\n{L('Product outcome you are researching for (e.g. «increase activation rate»): ', 'Product outcome для которого исследуешь (например «повысить активацию»): ')}"
    ).strip()

    print(f"\n{L('Add signals one by one. Empty line on source to finish.', 'Добавляй сигналы по одному. Пустая строка в источнике — завершить.')}")
    print(L(f"Sources: {', '.join(SIGNAL_SOURCES)}", f"Источники: {', '.join(SIGNAL_SOURCES)}"))

    new_signals = []
    idx = 1
    while True:
        print(f"\n{L('Signal', 'Сигнал')} {idx}")
        source = input(f"  {L('Source (or Enter to finish)', 'Источник (или Enter чтобы завершить)')}: ").strip()
        if not source:
            break
        if source not in SIGNAL_SOURCES:
            source = "other"
        text = input(f"  {L('Text (paste verbatim or summary)', 'Текст (вставь дословно или кратко)')}: ").strip()
        if not text:
            continue
        new_signals.append({"source": source, "text": text})
        idx += 1

    audit = log(state, "gather_signals",
                f"outcome={outcome} new_signals={len(new_signals)}")
    return {**state, "outcome": outcome, "new_signals": new_signals, "audit": audit}


def classify_signals(state: CRState) -> CRState:
    """Classify each new signal using LLM → map to opportunity."""
    if not state.get("new_signals"):
        audit = log(state, "classify", L("No new signals to classify", "Нет новых сигналов для классификации"))
        return {**state, "audit": audit}

    print(f"\n🏷️  {L('Classifying signals...', 'Классифицирую сигналы...')}")
    classified = []

    for sig in state["new_signals"]:
        result = call_model(CLASSIFY_PROMPT.format(
            lang=get_language_instruction(),
            text=sig["text"],
            source=sig["source"],
        ))
        opp = result.get("opportunity", "Unclassified")
        sentiment = result.get("sentiment", 0.0)
        tags = result.get("tags", [])

        save_signal(sig["source"], sig["text"], opp, sentiment, tags)
        classified.append({**sig, "opportunity": opp, "sentiment": sentiment,
                            "insight": result.get("insight", ""),
                            "assumption": result.get("assumption_to_test", "")})

        sentiment_icon = "😊" if sentiment > 0.2 else "😐" if sentiment > -0.2 else "😞"
        print(f"   {sentiment_icon} [{sig['source']}] → {opp[:50]}")

    # Check alerts
    all_recent = get_recent_signals(7)
    alerts = _check_alerts(all_recent)
    if alerts:
        print(f"\n🚨 {L('Alerts fired', 'Сработали алерты')}: {len(alerts)}")
        for a in alerts:
            print(f"   [{a['type'].upper()}] {a['message']}")

    audit = log(state, "classify",
                f"classified={len(classified)} alerts={len(alerts)}")
    return {**state,
            "new_signals": classified,
            "alerts_fired": alerts,
            "audit": audit}


def detect_patterns(state: CRState) -> CRState:
    """Detect patterns and anomalies across all recent signals."""
    print(f"\n🔬 {L('Detecting patterns...', 'Нахожу паттерны...')}")

    recent = get_recent_signals(7)
    top_opps = get_top_opportunities(8)

    if not recent:
        audit = log(state, "patterns", L("No signals in last 7 days", "Нет сигналов за 7 дней"))
        return {**state, "patterns": {}, "audit": audit}

    # Compact signals summary for prompt (token-efficient)
    signals_summary = "\n".join(
        f"- [{s['source']}] {s['opportunity'] or '?'}: {s['raw_text'][:80]}"
        for s in recent[:30]  # cap at 30 to save tokens
    )
    opps_summary = "\n".join(
        f"- {o['title']} ({o['signal_count']} signals, sentiment={o['avg_sentiment']:.1f})"
        for o in top_opps
    )

    patterns = call_model(PATTERN_PROMPT.format(
        lang=get_language_instruction(),
        n=len(recent),
        signals_summary=signals_summary,
        opportunities=opps_summary,
    ))

    n_patterns = len(patterns.get("patterns", []))
    n_prio = len(patterns.get("ost_priorities", []))
    print(f"   ✅ {n_patterns} {L('patterns', 'паттернов')} | "
          f"{n_prio} {L('OST priorities', 'приоритетов OST')} | "
          f"{L('confidence', 'уверенность')}: {patterns.get('confidence', '?')}")

    # Print top OST priorities
    for p in patterns.get("ost_priorities", [])[:3]:
        action = p.get("recommended_action", "?")
        strength = p.get("evidence_strength", "?")
        print(f"   📌 [{action.upper()}|{strength}] {p.get('opportunity', '')[:55]}")

    audit = log(state, "patterns", f"patterns={n_patterns} ost_priorities={n_prio}")
    return {**state, "patterns": patterns, "audit": audit}


def generate_digest(state: CRState) -> CRState:
    """Generate weekly discovery digest."""
    print(f"\n📋 {L('Generating weekly digest...', 'Генерирую еженедельный дайджест...')}")

    recent = get_recent_signals(7)
    top_opps = get_top_opportunities(5)
    alerts = get_unresolved_alerts()
    week_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    if not recent and not top_opps:
        audit = log(state, "digest", L("Insufficient data for digest", "Недостаточно данных для дайджеста"))
        return {**state, "digest": {}, "needs_human": False, "audit": audit}

    digest = call_model(DIGEST_PROMPT.format(
        lang=get_language_instruction(),
        week=f"{week_start} — {datetime.now().strftime('%Y-%m-%d')}",
        n_signals=get_db_stats()["total_signals"],
        new_signals=len(recent),
        opportunities=json.dumps([
            {"title": o["title"], "signals": o["signal_count"],
             "sentiment": round(o.get("avg_sentiment") or 0, 2)}
            for o in top_opps
        ], ensure_ascii=False),
        patterns=json.dumps(state.get("patterns", {}).get("patterns", [])[:4],
                             ensure_ascii=False),
        alerts=[a["message"] for a in alerts[:3]],
    ))

    headline = digest.get("headline", "")
    print(f"   ✅ {headline[:70]}")

    save_digest(week_start, json.dumps(digest, ensure_ascii=False), len(recent))

    audit = log(state, "digest", f"week={week_start} signals={len(recent)}")
    return {**state, "digest": digest, "needs_human": True, "audit": audit}


def human_checkpoint(state: CRState) -> CRState:
    """Show digest and ask for approval."""
    digest = state.get("digest", {})
    alerts = state.get("alerts_fired", [])
    patterns = state.get("patterns", {})

    print(f"\n{'=' * 60}")
    print(f"✋ {L('WEEKLY DIGEST — REVIEW', 'ЕЖЕНЕДЕЛЬНЫЙ ДАЙДЖЕСТ — ПРОВЕРКА')}")
    print(f"{'=' * 60}")

    # Headline
    headline = digest.get("headline", "—")
    print(f"\n💡 {headline}")

    # Signal summary
    print(f"\n📊 {L('This week', 'За неделю')}:")
    print(f"   {digest.get('signal_summary', '—')}")

    # Top opportunities
    opps = digest.get("top_opportunities", [])
    if opps:
        print(f"\n🎯 {L('Top opportunities', 'Топ возможностей')}:")
        for o in opps[:3]:
            print(f"   • {o.get('opportunity', '')} — {o.get('action', '')}")

    # Assumption to test
    assumption = digest.get("assumption_to_test", {})
    if assumption:
        method = assumption.get("test_method", "?")
        print(f"\n🔬 {L('Test this assumption', 'Проверь это допущение')}: "
              f"{assumption.get('assumption', '')} [{method}]")
        print(f"   {L('Question', 'Вопрос')}: {assumption.get('question', '')}")

    # Interview recommendation
    interview = digest.get("recommended_interview", {})
    if interview:
        print(f"\n👤 {L('Talk to', 'Поговори с')}: {interview.get('profile', '')} — "
              f"{interview.get('reason', '')}")

    # Alerts
    if alerts:
        print(f"\n🚨 {L('Alerts', 'Алерты')} ({len(alerts)}):")
        for a in alerts[:3]:
            print(f"   [{a['type'].upper()}] {a['message']}")

    # Planning risks
    risks = digest.get("planning_risks", [])
    if risks:
        print(f"\n⚠️  {L('Planning risks', 'Риски для планирования')}:")
        for r in risks[:3]:
            print(f"   • {r}")

    # OST priorities from patterns
    ost = patterns.get("ost_priorities", [])
    if ost:
        print(f"\n📌 {L('OST priorities', 'Приоритеты OST')}:")
        for p in ost[:4]:
            action = p.get("recommended_action", "?")
            print(f"   [{action.upper()}] {p.get('opportunity', '')[:55]}")

    print(f"\n{'=' * 60}")
    print(L("y — approve + save | n — skip save | s — show full digest",
            "y — утвердить + сохранить | n — не сохранять | s — показать полный дайджест"))
    choice = input(f"\n{L('Choice', 'Выбор')}: ").strip().lower()

    if choice == "s":
        print(f"\n{json.dumps(digest, ensure_ascii=False, indent=2)[:2000]}")
        choice = input(f"\n{L('y/n', 'y/n')}: ").strip().lower()

    audit = log(state, "checkpoint", f"choice={choice}")
    return {**state,
            "human_approved": choice == "y",
            "needs_human": False,
            "audit": audit}


def finalize(state: CRState) -> CRState:
    """Save digest file and audit log."""
    sid = state["session_id"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = state.get("digest", {})
    lang = get_language()

    if state.get("human_approved") and digest:
        # Markdown digest file
        week = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        if lang == "en":
            doc = f"""# Weekly Discovery Digest
Week: {week} — {datetime.now().strftime('%Y-%m-%d')} | Session: {sid}

## {digest.get('headline', '')}

{digest.get('signal_summary', '')}

## Top Opportunities

"""
            for o in digest.get("top_opportunities", []):
                doc += f"### {o.get('opportunity', '')}\n"
                doc += f"Evidence: {o.get('evidence', '')}\n"
                doc += f"Action: {o.get('action', '')}\n\n"

            assumption = digest.get("assumption_to_test", {})
            if assumption:
                doc += f"## Assumption to Test This Week\n\n"
                doc += f"**Assumption:** {assumption.get('assumption', '')}\n"
                doc += f"**Method:** {assumption.get('test_method', '')}\n"
                doc += f"**Question:** {assumption.get('question', '')}\n\n"

            interview = digest.get("recommended_interview", {})
            if interview:
                doc += f"## Recommended Interview\n\n"
                doc += f"**Profile:** {interview.get('profile', '')}\n"
                doc += f"**Reason:** {interview.get('reason', '')}\n"
                doc += f"**Question:** {interview.get('question', '')}\n\n"

            risks = digest.get("planning_risks", [])
            if risks:
                doc += "## Planning Risks\n\n"
                for r in risks:
                    doc += f"- {r}\n"
        else:
            doc = f"""# Еженедельный дайджест исследований
Неделя: {week} — {datetime.now().strftime('%Y-%m-%d')} | Сессия: {sid}

## {digest.get('headline', '')}

{digest.get('signal_summary', '')}

## Топ возможностей

"""
            for o in digest.get("top_opportunities", []):
                doc += f"### {o.get('opportunity', '')}\n"
                doc += f"Доказательства: {o.get('evidence', '')}\n"
                doc += f"Действие: {o.get('action', '')}\n\n"

            assumption = digest.get("assumption_to_test", {})
            if assumption:
                doc += "## Допущение для проверки на этой неделе\n\n"
                doc += f"**Допущение:** {assumption.get('assumption', '')}\n"
                doc += f"**Метод:** {assumption.get('test_method', '')}\n"
                doc += f"**Вопрос:** {assumption.get('question', '')}\n\n"

            interview = digest.get("recommended_interview", {})
            if interview:
                doc += "## Рекомендуемое интервью\n\n"
                doc += f"**Профиль:** {interview.get('profile', '')}\n"
                doc += f"**Причина:** {interview.get('reason', '')}\n"
                doc += f"**Вопрос:** {interview.get('question', '')}\n\n"

            risks = digest.get("planning_risks", [])
            if risks:
                doc += "## Риски для планирования\n\n"
                for r in risks:
                    doc += f"- {r}\n"

        digest_file = f"cr_digest_{ts}.md"
        Path(digest_file).write_text(doc, encoding="utf-8")
    else:
        digest_file = None

    # Audit
    audit_file = f"cr_audit_{sid}_{ts}.json"
    Path(audit_file).write_text(
        json.dumps(state.get("audit", []), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Summary
    stats = get_db_stats()
    print(f"\n{'=' * 60}")
    if lang == "en":
        print(f"✅ Session complete!")
        print(f"   {len(state.get('new_signals', []))} signals added | "
              f"{stats['total_signals']} total | "
              f"{stats['opportunities']} open opportunities")
        if digest_file:
            print(f"📋 Digest: {digest_file}")
    else:
        print(f"✅ Сессия завершена!")
        print(f"   {len(state.get('new_signals', []))} сигналов добавлено | "
              f"{stats['total_signals']} всего | "
              f"{stats['opportunities']} открытых возможностей")
        if digest_file:
            print(f"📋 Дайджест: {digest_file}")

    audit = log(state, "finalize",
                f"signals={len(state.get('new_signals', []))} digest={digest_file}")
    return {**state, "audit": audit}


# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────

def after_checkpoint(state: CRState) -> Literal["finalize"]:
    return "finalize"


# ─────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────

def build_graph(db_path: str = "cr_checkpoints.db"):
    g = StateGraph(CRState)
    for name, fn in [
        ("gather_signals",  gather_signals),
        ("classify_signals", classify_signals),
        ("detect_patterns", detect_patterns),
        ("generate_digest", generate_digest),
        ("human_checkpoint", human_checkpoint),
        ("finalize",        finalize),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("gather_signals")
    g.add_edge("gather_signals",   "classify_signals")
    g.add_edge("classify_signals", "detect_patterns")
    g.add_edge("detect_patterns",  "generate_digest")
    g.add_edge("generate_digest",  "human_checkpoint")
    g.add_conditional_edges("human_checkpoint", after_checkpoint, {"finalize": "finalize"})
    g.add_edge("finalize", END)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    return g.compile(checkpointer=SqliteSaver(conn))


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

def run_cr(
    outcome: str = "",
    digest_only: bool = False,
    session_id: str = None,
    resume: bool = False,
) -> tuple[dict, str]:
    init_db()
    sid = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    app = build_graph()
    config = {"configurable": {"thread_id": sid}}

    if resume:
        print(f"\n▶️  {L('Resuming session', 'Продолжаю сессию')} {sid}...")
        final = app.invoke(None, config=config)
    else:
        initial = CRState(
            digest_only=digest_only,
            outcome=outcome,
            new_signals=[],
            patterns={},
            digest={},
            alerts_fired=[],
            needs_human=False,
            human_approved=False,
            session_id=sid,
            audit=[],
        )
        final = app.invoke(initial, config=config)

    print(f"\n💾 Session ID: {sid}")
    return final, sid


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lang", choices=["en", "ru"], default=None)
    parser.add_argument("--digest", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args, _ = parser.parse_known_args()
    run_cr(digest_only=args.digest, session_id=args.resume, resume=bool(args.resume))
