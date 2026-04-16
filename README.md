# 🔍 Continuous Research Agent

AI-powered continuous discovery agent built with LangGraph + Claude. Based on Teresa Torres' Continuous Discovery Habits framework. Monitors product signals between sprints so discovery never stops.

## What it does

1. **Signal Collector** — accepts signals from any source: user interviews, support tickets, app reviews, NPS comments, sales calls, analytics, or other. Each signal is classified and mapped to an opportunity on the OST.
2. **OST Maintainer** — maintains an Opportunity Solution Tree in SQLite. Every signal increments the opportunity's count and updates its average sentiment automatically.
3. **Pattern Detector** — analyzes signals from the past 7 days to find emerging themes, anomalies, and challenged assumptions. Prioritizes OST opportunities by evidence strength (`strong / moderate / weak`) and recommended action (`explore / test / monitor / close`).
4. **Weekly Digest Generator** — produces a structured weekly report: headline, signal summary, top 3 opportunities with evidence, one assumption to test this week, one specific interview recommendation with profile and question.
5. **Alert Engine** — fires automatically when signal volume on one opportunity crosses threshold (5+ this week) or when 3+ strongly negative signals appear.

## How it works

```
PM adds signals (source + text)
        ↓
LLM classifies → opportunity title, sentiment, tags, insight, assumption
        ↓
Signals saved to DB + OST updated
        ↓
Alert engine checks thresholds
        ↓
Pattern detection across last 7 days
        ↓
Weekly digest generated (headline → opportunities → assumption → interview)
        ↓
Human checkpoint — approve or skip
        ↓
Digest saved as markdown + audit log
```

## Two modes

```bash
# Regular mode — add signals + generate digest
python3 cr_agent.py --lang en

# Digest-only mode — skip signal collection, just generate weekly report
python3 cr_agent.py --lang en --digest
```

## Quick start

```bash
git clone https://github.com/artemsanisimow-ux/cr-agent.git
cd cr-agent
python3 -m venv venv
source venv/bin/activate
pip install langgraph langchain-anthropic langgraph-checkpoint-sqlite python-dotenv
```

Add to `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
LANGUAGE=en
```

```bash
python3 cr_agent.py --lang en
```

## Signal sources

| Source | When to use |
|--------|-------------|
| `user_interview` | Verbatim quotes or summaries from user sessions |
| `support_ticket` | Customer support issues and requests |
| `app_review` | App Store / Play Store / marketplace reviews |
| `nps_comment` | NPS survey open-ended responses |
| `sales_call` | Objections and requests heard on sales calls |
| `analytics` | Metric drops, funnel anomalies, cohort findings |
| `other` | Any other signal source |

## Alert thresholds

| Alert | Condition |
|-------|-----------|
| Volume | Same opportunity mentioned 5+ times in 7 days |
| Sentiment | 3+ strongly negative signals (sentiment < -0.5) in 7 days |

## Output

- `cr_digest_TIMESTAMP.md` — weekly digest in markdown
- `cr_audit_SESSION_TIMESTAMP.json` — step-by-step log
- `cr_signals.db` — SQLite with 4 tables: signals, opportunities, digests, alerts

## Database schema

```sql
signals       — raw signal texts with source, opportunity, sentiment, tags
opportunities — OST nodes with signal_count and avg_sentiment (auto-updated)
digests       — weekly digest history
alerts        — fired alerts with resolved flag
```

## Running tests

```bash
pytest test_cr_agent.py -v   # 49 tests
```

Tests cover: DB schema, CRUD operations, alert thresholds, signal classification, pattern detection, digest generation, i18n, graph structure.

## Part of a larger system

| Agent | Repo | Description |
|-------|------|-------------|
| Discovery | [discovery-agent](https://github.com/artemsanisimow-ux/discovery-agent) | Raw data → insights → hypotheses |
| Grooming | [grooming-agent](https://github.com/artemsanisimow-ux/grooming-agent) | Jira + Linear → estimate → acceptance criteria |
| Planning | [planning-agent](https://github.com/artemsanisimow-ux/planning-agent) | Monte Carlo → sprint plan → publish |
| Retrospective | [retro-agent](https://github.com/artemsanisimow-ux/retro-agent) | Sprint analysis → action items → planning insights |
| PRD | [prd-agent](https://github.com/artemsanisimow-ux/prd-agent) | Feature → full PRD → tasks in Jira + Linear |
| Stakeholder | [stakeholder-agent](https://github.com/artemsanisimow-ux/stakeholder-agent) | Sprint data → tailored comms for 5 audiences |
| A/B Testing | [ab-agent](https://github.com/artemsanisimow-ux/ab-agent) | Hypothesis → experiment design → ship/iterate/kill |
| Continuous Research | this repo | Signals → OST → weekly digest → alerts |
