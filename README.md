# RegimePilot (practice project)

Practice project for the **Alpaca AI Trading Agents Hackathon** (Aug 28 - Sep 4, 2026).

> **Disposable.** Until the organizers confirm whether pre-kickoff code is allowed,
> treat this folder as practice, not the official submission. The judged submission
> must use a **fresh Alpaca paper account funded with exactly $100,000**.

## Status: Phase 3 complete

Phase 3 adds read-only AI direction proposals (`BUY_CALL` / `BUY_PUT` / `HOLD`).
No orders are submitted yet.

| # | Phase | State |
|---|---------------------------------------|-------------|
| 1 | Environment and read-only connectivity | done |
| 2 | Read-only market observer + features | done |
| 3 | AI trade proposal, no execution | **current** |
| 4 | Deterministic contract selector + risk gate | not started |
| 5 | Dry-run order generation | not started |
| 6 | Small paper options trade | not started |
| 7 | Autonomous 15-minute loop | not started |
| 8 | Dashboard and hackathon submission | not started |

## Safety rules this code enforces

- Credentials are read from **environment variables only**.
- `ALPACA_PAPER` defaults to `true`, and an unrecognised value is an error, not a guess.
- Startup **aborts** if a live-trading flag (`ALPACA_LIVE`, `ALPACA_LIVE_TRADING`,
  `APCA_LIVE`) is true, or if any endpoint variable points at `api.alpaca.markets`.
- `TradingClient` is always constructed with `paper=True` hard-coded, so no
  environment value can flip it to live.
- Credentials live in `SecretStr` and are never printed or logged.
- There is **no** function to submit, cancel or replace an order, and none to close
  or exercise a position. Do not add one before the phase that calls for it.
- Unit tests make **no** network calls.

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

### Create your `.env` yourself

`.env` is git-ignored and **must never be committed or pasted into a chat**.

1. Copy the template:

   ```bash
   cp .env.example .env      # PowerShell: Copy-Item .env.example .env
   ```

2. Open `.env` in an editor and paste your **paper** keys from
   <https://app.alpaca.markets/paper/dashboard/overview>:

   ```dotenv
   ALPACA_API_KEY=PK...your paper key...
   ALPACA_SECRET_KEY=...your paper secret...
   ALPACA_PAPER=true
   ```

3. Leave `ALPACA_PAPER=true`. Any other value stops the program.

4. For live LLM decisions (optional in Phase 3), add an OpenRouter key from
   <https://openrouter.ai/>:

   ```dotenv
   OPENROUTER_API_KEY=sk-or-...
   ```

   Use `--stub` if you do not have an OpenRouter key yet.

Do not set `ALPACA_BASE_URL` or `APCA_API_BASE_URL`. If you do, they must point at
`paper-api.alpaca.markets`; a live URL aborts startup.

## Run

Mocked unit tests (no credentials and no network needed):

```bash
uv run pytest
```

Read-only connectivity check (needs a filled-in `.env`):

```bash
uv run python -m regimepilot.smoke_test
```

Phase 2 feature observation:

```bash
uv run python -m regimepilot.history
uv run python -m regimepilot.history --json
```

Phase 3 modules:

```bash
# Filtered Alpaca news for SPY
uv run python -m regimepilot.news --json

# Full LLM briefing (features + news + pre-gates)
uv run python -m regimepilot.evidence --json

# Trade direction proposal
uv run python -m regimepilot.decision --stub --json   # no OpenRouter key needed
uv run python -m regimepilot.decision --json          # calls GLM-5.3 Flash via OpenRouter, free chain as fallback
```

`decision --json` prints one `TradeProposal`:

```json
{
  "observed_at": "2026-08-25T14:30:00+00:00",
  "symbol": "SPY",
  "action": "BUY_CALL",
  "confidence": "medium",
  "thesis": "Stub rule: 15m and 60m momentum align upward.",
  "evidence_used": ["gates.momentum_align", "underlying.return_15m", "underlying.return_60m"],
  "gate_skipped": false,
  "model": "stub"
}
```

Pre-gate failures return `action: "HOLD"` with `gate_skipped: true` without calling
the LLM.

## Layout

```text
.
├── .env.example
├── pyproject.toml
├── README.md
├── src/regimepilot/
│   ├── config.py         # credential loading + paper-trading guards
│   ├── smoke_test.py     # Phase 1 connectivity check
│   ├── models.py         # frozen observation models
│   ├── observer.py       # Phase 2A read-only market observer
│   ├── features.py       # Phase 2B deterministic features
│   ├── history.py        # Phase 2B Alpaca bar/quote reads
│   ├── gates.py          # Phase 3A pre-gates + session labels
│   ├── news.py           # Phase 3B filtered Alpaca news
│   ├── evidence.py       # Phase 3C evidence briefing assembly
│   └── decision.py       # Phase 3D LLM / stub trade proposal
└── tests/
    ├── test_config.py
    ├── test_smoke_test.py
    ├── test_observer.py
    ├── test_features.py
    ├── test_history.py
    ├── test_gates.py
    ├── test_news.py
    ├── test_evidence.py
    └── test_decision.py
```

## Planned baseline (do not change without approval)

Monitor SPY only. Gather market data, option contracts, chain data and account
state. Emit one of `BUY_CALL`, `BUY_PUT`, `HOLD`. Deterministic code (not the LLM)
picks the exact contract and quantity. Every proposal passes hard risk checks.
Execution is Alpaca paper only. Every decision is logged, including `HOLD` and
rejected trades.

Phase 3 now includes filtered Alpaca news as LLM context. Out of scope for now:
multi-agent architectures, reinforcement learning, Jump or Hidden Markov Models,
vertical spreads and multi-leg options, 0DTE, multiple underlyings, automatic
strategy optimization.
