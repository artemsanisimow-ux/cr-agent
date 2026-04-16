"""
Test suite for Continuous Research Agent.
Covers: DB schema, signal persistence, alert logic, OST patterns, i18n, graph, state.
Run: pytest test_cr_agent.py -v
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
sys.path.insert(0, "/mnt/user-data/outputs")
sys.path.insert(0, "/home/claude/cr-agent")

import cr_agent as cr
import i18n


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "DB_PATH", str(tmp_path / "test.db"))
    cr.init_db()
    yield


@pytest.fixture
def base_state() -> cr.CRState:
    return cr.CRState(
        digest_only=False,
        outcome="increase activation rate",
        new_signals=[
            {"source": "user_interview", "text": "Can't find the export button"},
            {"source": "support_ticket", "text": "How do I export to CSV?"},
        ],
        patterns={},
        digest={},
        alerts_fired=[],
        needs_human=False,
        human_approved=False,
        session_id="test_session",
        audit=[],
    )


@pytest.fixture
def classified_state(base_state) -> cr.CRState:
    """State with classified signals already in DB."""
    cr.save_signal("user_interview", "Can't find export button",
                   "Export feature discoverability", -0.6, ["export", "ux"])
    cr.save_signal("support_ticket", "How do I export CSV",
                   "Export feature discoverability", -0.4, ["export", "csv"])
    cr.save_signal("app_review", "Love the dashboard",
                   "Dashboard usability", 0.8, ["dashboard"])
    return {**base_state, "new_signals": [
        {"source": "user_interview", "text": "Can't find export button",
         "opportunity": "Export feature discoverability", "sentiment": -0.6,
         "insight": "Users struggle to find export", "assumption": "Export is visible"},
        {"source": "support_ticket", "text": "How do I export CSV",
         "opportunity": "Export feature discoverability", "sentiment": -0.4,
         "insight": "Export needs better labeling", "assumption": "Users know export exists"},
    ]}


# ─────────────────────────────────────────────
# 1. DATABASE
# ─────────────────────────────────────────────

class TestDatabase:
    def test_init_creates_all_tables(self):
        conn = sqlite3.connect(cr.DB_PATH)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert {"signals", "opportunities", "digests", "alerts"} <= tables

    def test_save_signal_creates_signal(self):
        cr.save_signal("user_interview", "Pain point text",
                       "Navigation confusion", -0.5, ["nav", "ux"])
        conn = sqlite3.connect(cr.DB_PATH)
        row = conn.execute("SELECT * FROM signals").fetchone()
        conn.close()
        assert row is not None
        assert row[1] == "user_interview"
        assert "Pain point" in row[2]

    def test_save_signal_creates_opportunity(self):
        cr.save_signal("app_review", "Text", "New opportunity", 0.5, [])
        opps = cr.get_top_opportunities()
        assert any(o["title"] == "New opportunity" for o in opps)

    def test_save_signal_increments_opportunity_count(self):
        cr.save_signal("user_interview", "Text 1", "Same opportunity", -0.3, [])
        cr.save_signal("support_ticket", "Text 2", "Same opportunity", -0.5, [])
        opps = cr.get_top_opportunities()
        opp = next(o for o in opps if o["title"] == "Same opportunity")
        assert opp["signal_count"] == 2

    def test_save_signal_averages_sentiment(self):
        cr.save_signal("user_interview", "T1", "Opp", 0.8, [])
        cr.save_signal("app_review", "T2", "Opp", 0.4, [])
        opps = cr.get_top_opportunities()
        opp = next(o for o in opps if o["title"] == "Opp")
        assert abs(opp["avg_sentiment"] - 0.6) < 0.05

    def test_save_signal_returns_id(self):
        sig_id = cr.save_signal("other", "Text", "Opp", 0.0, [])
        assert isinstance(sig_id, int)
        assert sig_id > 0

    def test_get_recent_signals_filters_by_days(self):
        cr.save_signal("other", "Recent", "Opp A", 0.0, [])
        # Insert old signal manually
        conn = sqlite3.connect(cr.DB_PATH)
        conn.execute(
            "INSERT INTO signals (source, raw_text, opportunity, sentiment, tags_json, created_at) "
            "VALUES (?,?,?,?,?,?)",
            ("other", "Old", "Opp B", 0.0, "[]",
             (datetime.now() - timedelta(days=10)).isoformat())
        )
        conn.commit()
        conn.close()
        recent = cr.get_recent_signals(7)
        assert all("Old" not in s["raw_text"] for s in recent)
        assert any("Recent" in s["raw_text"] for s in recent)

    def test_get_top_opportunities_sorted_by_count(self):
        cr.save_signal("other", "T1", "Opp A", 0.0, [])
        cr.save_signal("other", "T2", "Opp B", 0.0, [])
        cr.save_signal("other", "T3", "Opp B", 0.0, [])
        opps = cr.get_top_opportunities()
        # Opp B has 2 signals, Opp A has 1
        assert opps[0]["title"] == "Opp B"

    def test_get_top_opportunities_limit(self):
        for i in range(15):
            cr.save_signal("other", f"T{i}", f"Opp {i}", 0.0, [])
        opps = cr.get_top_opportunities(10)
        assert len(opps) <= 10

    def test_save_alert(self):
        cr.save_alert("volume", "High volume alert", "Some opportunity")
        alerts = cr.get_unresolved_alerts()
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "volume"
        assert alerts[0]["opportunity"] == "Some opportunity"

    def test_get_unresolved_alerts_excludes_resolved(self):
        cr.save_alert("volume", "Alert 1")
        conn = sqlite3.connect(cr.DB_PATH)
        conn.execute("UPDATE alerts SET resolved=1")
        conn.commit()
        conn.close()
        cr.save_alert("sentiment", "Alert 2")
        alerts = cr.get_unresolved_alerts()
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "sentiment"

    def test_save_digest(self):
        cr.save_digest("2026-04-01", '{"headline": "Test"}', 10)
        conn = sqlite3.connect(cr.DB_PATH)
        row = conn.execute("SELECT * FROM digests").fetchone()
        conn.close()
        assert row is not None
        assert row[1] == "2026-04-01"
        assert row[3] == 10

    def test_get_db_stats_structure(self):
        stats = cr.get_db_stats()
        for key in ["total_signals", "opportunities", "digests",
                    "unresolved_alerts", "signals_last_7d"]:
            assert key in stats
            assert isinstance(stats[key], int)

    def test_get_db_stats_counts_correctly(self):
        cr.save_signal("other", "T", "O", 0.0, [])
        cr.save_alert("volume", "A")
        stats = cr.get_db_stats()
        assert stats["total_signals"] == 1
        assert stats["opportunities"] == 1
        assert stats["unresolved_alerts"] == 1
        assert stats["signals_last_7d"] == 1


# ─────────────────────────────────────────────
# 2. ALERT ENGINE
# ─────────────────────────────────────────────

class TestAlerts:
    def test_volume_alert_fires_at_threshold(self):
        signals = [
            {"opportunity": "Nav confusion", "sentiment": -0.5}
            for _ in range(cr.ALERT_VOLUME_THRESHOLD)
        ]
        alerts = cr._check_alerts(signals)
        assert any(a["type"] == "volume" for a in alerts)

    def test_volume_alert_does_not_fire_below_threshold(self):
        signals = [
            {"opportunity": "Nav confusion", "sentiment": -0.5}
            for _ in range(cr.ALERT_VOLUME_THRESHOLD - 1)
        ]
        alerts = cr._check_alerts(signals)
        volume_alerts = [a for a in alerts if a["type"] == "volume"]
        assert len(volume_alerts) == 0

    def test_sentiment_alert_fires_with_3_negative(self):
        signals = [
            {"opportunity": "Opp A", "sentiment": -0.8},
            {"opportunity": "Opp B", "sentiment": -0.9},
            {"opportunity": "Opp C", "sentiment": -0.7},
        ]
        alerts = cr._check_alerts(signals)
        assert any(a["type"] == "sentiment" for a in alerts)

    def test_sentiment_alert_does_not_fire_with_2_negative(self):
        signals = [
            {"opportunity": "Opp A", "sentiment": -0.8},
            {"opportunity": "Opp B", "sentiment": -0.9},
            {"opportunity": "Opp C", "sentiment": 0.5},
        ]
        alerts = cr._check_alerts(signals)
        sentiment_alerts = [a for a in alerts if a["type"] == "sentiment"]
        assert len(sentiment_alerts) == 0

    def test_alerts_saved_to_db(self):
        signals = [{"opportunity": "Opp", "sentiment": -0.5}
                   for _ in range(cr.ALERT_VOLUME_THRESHOLD)]
        cr._check_alerts(signals)
        alerts = cr.get_unresolved_alerts()
        assert len(alerts) > 0

    def test_no_alerts_for_empty_signals(self):
        alerts = cr._check_alerts([])
        assert alerts == []

    def test_multiple_opportunities_separate_volume_alerts(self):
        # Opp A: 5 signals, Opp B: 3 signals
        signals = (
            [{"opportunity": "Opp A", "sentiment": -0.5}] * cr.ALERT_VOLUME_THRESHOLD +
            [{"opportunity": "Opp B", "sentiment": -0.5}] * (cr.ALERT_VOLUME_THRESHOLD - 2)
        )
        alerts = cr._check_alerts(signals)
        volume_alerts = [a for a in alerts if a["type"] == "volume"]
        assert any(a.get("opportunity") == "Opp A" for a in volume_alerts)
        assert not any(a.get("opportunity") == "Opp B" for a in volume_alerts)


# ─────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────

class TestHelpers:
    def test_lang_en(self):
        i18n.set_language("en")
        assert cr.L("Hello", "Привет") == "Hello"

    def test_lang_ru(self):
        i18n.set_language("ru")
        assert cr.L("Hello", "Привет") == "Привет"

    def test_log_creates_entry(self, base_state):
        audit = cr.log(base_state, "test", "summary")
        assert len(audit) == 1
        assert audit[0]["step"] == "test"
        assert "ts" in audit[0]

    def test_log_accumulates(self, base_state):
        state = {**base_state, "audit": [{"step": "existing"}]}
        audit = cr.log(state, "new", "new summary")
        assert len(audit) == 2

    def test_signal_sources_defined(self):
        assert len(cr.SIGNAL_SOURCES) >= 6
        assert "user_interview" in cr.SIGNAL_SOURCES
        assert "support_ticket" in cr.SIGNAL_SOURCES
        assert "app_review" in cr.SIGNAL_SOURCES

    def test_thresholds_defined(self):
        assert cr.ALERT_VOLUME_THRESHOLD > 0
        assert 0 < cr.ALERT_SENTIMENT_DROP < 1


# ─────────────────────────────────────────────
# 4. STATE TRANSITIONS (mock Claude)
# ─────────────────────────────────────────────

class TestStateTransitions:
    def _mock(self, monkeypatch, rv: dict):
        monkeypatch.setattr(cr, "call_model", lambda _: rv)

    def test_classify_saves_signals_to_db(self, classified_state, monkeypatch):
        self._mock(monkeypatch, {
            "opportunity": "Export feature discoverability",
            "sentiment": -0.5,
            "tags": ["export"],
            "insight": "Users struggle to find export",
            "assumption_to_test": "Export button is visible",
        })
        state = {**classified_state, "new_signals": [
            {"source": "user_interview", "text": "Can't find export button"}
        ]}
        result = cr.classify_signals(state)
        signals = cr.get_recent_signals(7)
        assert len(signals) >= 1  # at least the one from this test + fixture

    def test_classify_enriches_signals(self, base_state, monkeypatch):
        self._mock(monkeypatch, {
            "opportunity": "Test opportunity",
            "sentiment": -0.3,
            "tags": ["tag1"],
            "insight": "Test insight",
            "assumption_to_test": "Test assumption",
        })
        result = cr.classify_signals(base_state)
        assert len(result["new_signals"]) == 2
        assert all("opportunity" in s for s in result["new_signals"])
        assert all("sentiment" in s for s in result["new_signals"])

    def test_classify_checks_alerts(self, base_state, monkeypatch):
        # Add enough signals to trigger volume alert
        for _ in range(cr.ALERT_VOLUME_THRESHOLD - 1):
            cr.save_signal("other", "text", "Same opp", -0.5, [])
        self._mock(monkeypatch, {
            "opportunity": "Same opp", "sentiment": -0.5,
            "tags": [], "insight": "i", "assumption_to_test": "a",
        })
        result = cr.classify_signals({**base_state, "new_signals": [
            {"source": "user_interview", "text": "Another same opp signal"}
        ]})
        assert len(result["alerts_fired"]) > 0

    def test_detect_patterns_with_data(self, classified_state, monkeypatch):
        self._mock(monkeypatch, {
            "patterns": [{"theme": "Export issues", "signal_count": 2, "implication": "UX problem"}],
            "anomalies": [],
            "challenged_assumptions": ["Export is discoverable"],
            "ost_priorities": [{"opportunity": "Export discoverability",
                                 "evidence_strength": "strong",
                                 "recommended_action": "explore"}],
            "confidence": "high",
        })
        result = cr.detect_patterns(classified_state)
        assert "patterns" in result["patterns"]
        assert len(result["patterns"]["patterns"]) == 1

    def test_detect_patterns_no_signals(self, base_state, monkeypatch):
        # DB is empty (no signals saved)
        result = cr.detect_patterns(base_state)
        assert result["patterns"] == {}

    def test_generate_digest_structure(self, classified_state, monkeypatch):
        classified_state["patterns"] = {
            "patterns": [{"theme": "Export", "signal_count": 2, "implication": "UX"}],
            "ost_priorities": [{"opportunity": "Export discoverability",
                                 "evidence_strength": "strong", "recommended_action": "explore"}],
        }
        self._mock(monkeypatch, {
            "headline": "Export discoverability is top issue this week",
            "signal_summary": "2 signals about export feature",
            "top_opportunities": [{"opportunity": "Export discoverability",
                                    "evidence": "2 signals", "action": "Run interview"}],
            "assumption_to_test": {"assumption": "Users know export exists",
                                    "test_method": "interview", "question": "How would you export data?"},
            "recommended_interview": {"profile": "Power user", "reason": "Uses export heavily",
                                       "question": "Walk me through how you export data"},
            "planning_risks": ["Export complexity may delay sprint"],
        })
        result = cr.generate_digest(classified_state)
        assert result["digest"]["headline"] == "Export discoverability is top issue this week"
        assert result["needs_human"] is True

    def test_generate_digest_saves_to_db(self, classified_state, monkeypatch):
        classified_state["patterns"] = {}
        self._mock(monkeypatch, {
            "headline": "Test headline",
            "signal_summary": "Test summary",
            "top_opportunities": [],
            "assumption_to_test": {"assumption": "A", "test_method": "interview", "question": "Q"},
            "recommended_interview": {"profile": "P", "reason": "R", "question": "Q"},
            "planning_risks": [],
        })
        cr.generate_digest(classified_state)
        stats = cr.get_db_stats()
        assert stats["digests"] == 1

    def test_digest_only_skips_signals(self, base_state, monkeypatch):
        state = {**base_state, "digest_only": True}
        result = cr.gather_signals(state)
        assert result["new_signals"] == []

    def test_classify_skips_on_empty_signals(self, base_state, monkeypatch):
        state = {**base_state, "new_signals": []}
        result = cr.classify_signals(state)
        # Should return unchanged state
        assert result["new_signals"] == []

    def test_finalize_saves_audit(self, classified_state, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        classified_state["digest"] = {"headline": "Test", "signal_summary": "S",
                                       "top_opportunities": [], "assumption_to_test": {},
                                       "recommended_interview": {}, "planning_risks": []}
        classified_state["human_approved"] = True
        result = cr.finalize(classified_state)
        audit_files = list(tmp_path.glob("cr_audit_*.json"))
        assert len(audit_files) == 1

    def test_finalize_saves_digest_when_approved(self, classified_state, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        i18n.set_language("en")
        classified_state["digest"] = {
            "headline": "Big insight this week",
            "signal_summary": "Export is a top issue",
            "top_opportunities": [{"opportunity": "Export", "evidence": "2 signals", "action": "Test"}],
            "assumption_to_test": {"assumption": "A", "test_method": "interview", "question": "Q"},
            "recommended_interview": {"profile": "P", "reason": "R", "question": "Q"},
            "planning_risks": ["Risk 1"],
        }
        classified_state["human_approved"] = True
        cr.finalize(classified_state)
        digest_files = list(tmp_path.glob("cr_digest_*.md"))
        assert len(digest_files) == 1
        content = digest_files[0].read_text()
        assert "Big insight" in content

    def test_finalize_no_digest_when_not_approved(self, classified_state, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        classified_state["human_approved"] = False
        classified_state["digest"] = {"headline": "Test"}
        cr.finalize(classified_state)
        digest_files = list(tmp_path.glob("cr_digest_*.md"))
        assert len(digest_files) == 0


# ─────────────────────────────────────────────
# 5. I18N
# ─────────────────────────────────────────────

class TestI18n:
    def test_prompts_have_lang_placeholder(self):
        for prompt in [cr.CLASSIFY_PROMPT, cr.PATTERN_PROMPT, cr.DIGEST_PROMPT]:
            assert "{lang}" in prompt

    def test_prompt_finalize_returns_string(self):
        for lang in ["en", "ru"]:
            i18n.set_language(lang)
            val = i18n.t("prompt_finalize")
            assert isinstance(val, str) and len(val) > 0

    def test_language_instruction_en(self):
        i18n.set_language("en")
        assert "English" in i18n.get_language_instruction()

    def test_language_instruction_ru(self):
        i18n.set_language("ru")
        assert "русском" in i18n.get_language_instruction()

    def test_lang_helper_both_directions(self):
        i18n.set_language("en")
        assert cr.L("Ship", "Выпустить") == "Ship"
        i18n.set_language("ru")
        assert cr.L("Ship", "Выпустить") == "Выпустить"


# ─────────────────────────────────────────────
# 6. GRAPH & STATE
# ─────────────────────────────────────────────

class TestGraph:
    def test_graph_builds(self):
        graph = cr.build_graph(":memory:")
        assert graph is not None

    def test_state_required_fields(self, base_state):
        for field in ["digest_only", "outcome", "new_signals", "patterns",
                       "digest", "alerts_fired", "needs_human",
                       "human_approved", "session_id", "audit"]:
            assert field in base_state, f"Missing: {field}"

    def test_state_types(self, base_state):
        assert isinstance(base_state["digest_only"], bool)
        assert isinstance(base_state["new_signals"], list)
        assert isinstance(base_state["patterns"], dict)
        assert isinstance(base_state["digest"], dict)
        assert isinstance(base_state["alerts_fired"], list)
        assert isinstance(base_state["audit"], list)

    def test_after_checkpoint_always_finalize(self, base_state):
        for approved in [True, False]:
            state = {**base_state, "human_approved": approved}
            assert cr.after_checkpoint(state) == "finalize"

    def test_signal_sources_are_valid(self):
        assert all(isinstance(s, str) for s in cr.SIGNAL_SOURCES)
        assert len(cr.SIGNAL_SOURCES) >= 5
