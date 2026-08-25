# RegimePilot (practice project)

Practice project for the **Alpaca AI Trading Agents Hackathon** (Aug 28 - Sep 4, 2026).

> **Disposable.** Until the organizers confirm whether pre-kickoff code is allowed,
> treat this folder as practice, not the official submission. The judged submission
> must use a **fresh Alpaca paper account funded with exactly $100,000**.

## Status: Phase 1 only

Phase 1 is environment setup and read-only connectivity. Nothing else exists yet.

| # | Phase | State |
|---|---------------------------------------|-------------|
| 1 | Environment and read-only connectivity | **current** |
| 2 | Read-only market observer              | not started |
| 3 | AI trade proposal, no execution        | not started |
| 4 | Deterministic contract selector + risk gate | not started |
| 5 | Dry-run order generation               | not started |
| 6 | Small paper options trade              | not started |
| 7 | Autonomous 15-minute loop              | not started |
| 8 | Dashboard and hackathon submission     | not started |

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

It prints one JSON object and nothing else on stdout:

```json
{
  "timestamp": "2026-08-25T14:30:00+00:00",
  "market_open": true,
  "account_id_masked": "****8888",
  "spy_bar_count": 7,
  "spy_option_contract_count": 500,
  "earliest_expiration": "2026-08-28",
  "latest_expiration": "2026-09-08",
  "checks": {
    "config": "ok",
    "clock": "ok",
    "account": "ok",
    "spy_bars": "ok",
    "spy_option_contracts": "ok"
  }
}
```

Check statuses are `ok`, `empty` (call succeeded, returned nothing), `error`, or
`skipped`. On `error`, only the exception *type* is written to stderr, never its
message, because HTTP client errors can quote the request that produced them.

`spy_option_contract_count` counts the **first page** of active SPY contracts
expiring in 3-14 days (page limit 500), which is enough to prove connectivity.
`earliest_expiration` and `latest_expiration` describe only the contracts returned.

## Layout

```text
.
├── .env.example          # placeholders only, safe to commit
├── .gitignore            # keeps .env out of Git
├── .python-version       # pins uv to Python 3.11
├── pyproject.toml
├── README.md
├── src/regimepilot/
│   ├── __init__.py
│   ├── config.py         # credential loading + paper-trading guards
│   └── smoke_test.py     # read-only connectivity check
└── tests/
    ├── test_config.py
    └── test_smoke_test.py
```

## Planned baseline (do not change without approval)

Monitor SPY only. Gather market data, option contracts, chain data and account
state. Emit one of `BUY_CALL`, `BUY_PUT`, `HOLD`. Deterministic code (not the LLM)
picks the exact contract and quantity. Every proposal passes hard risk checks.
Execution is Alpaca paper only. Every decision is logged, including `HOLD` and
rejected trades.

Out of scope for now: multi-agent architectures, reinforcement learning, Jump or
Hidden Markov Models, vertical spreads and multi-leg options, 0DTE, news and
sentiment trading, multiple underlyings, automatic strategy optimization.
