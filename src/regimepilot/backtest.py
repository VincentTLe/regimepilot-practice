"""Replay historical SPY bars through the live decision pipeline and simulate fills.

This module answers one question before any strategy change reaches paper
trading: on past data, did this pipeline's decisions make money on a
risk-adjusted basis, compared to doing nothing or to naive momentum? It is a
checkpoint a strategy passes through on its way to a paper run, not a second
trading agent: see ``score.py`` for the scorecard that turns its output into a
pass/fail number.

Design choice -- reuse, don't reimplement: the backtest calls the exact same
pure functions the live pipeline calls (``features.build_feature_packet``,
``gates.evaluate_gates``, ``regime.classify_regime``, and either
``decision.stub_proposal`` or a supplied ``decision_fn``). Nothing about the
decision logic is duplicated here, so a backtest result reflects the actual
pipeline, not a parallel approximation of it that could quietly drift from
the code that will really trade.

What this module does NOT have and therefore approximates:

* No historical SPY option chain. Contract prices are simulated with
  ``black_scholes.option_price`` using the historical spot, a strike chosen
  the same way ``selector.py`` chooses one (nearest to spot), a fixed 7
  calendar-day expiration, and a volatility assumption. This means simulated
  fills ignore bid/ask spread, quote staleness, and the real (skewed) implied
  volatility surface -- a live-quote-based validation from actual paper
  fills is still required afterward and is expected to differ. This is the
  same limitation described in the project's regime-vs-IV-rank discussion:
  IV rank needs real historical option IV, which is unavailable here, so
  ``iv_rank`` on every ``RegimeReading`` produced by this module is always
  ``None``.
* No multi-day position management. A simulated entry is closed at the
  close of the same session it opened in (Simplification #1, see
  ``SimulatedTrade``). The live portfolio agent can hold a position for
  several sessions and decide HOLD/CLOSE on it each cycle; extending this
  backtest to do the same is future work, not implemented here.
* At most one open simulated position at a time, matching the live runner's
  "one new entry per cycle" rule but not its "up to three open positions"
  ceiling -- a simplification to keep one linear pass through history
  sufficient, rather than tracking a small portfolio.

None of this module submits, reads, or requires an Alpaca or OpenRouter
credential. It is pure once handed a bar history, exactly like
``features.py`` and ``regime.py``.
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta

from regimepilot import black_scholes
from regimepilot.console import tolerant_console
from regimepilot.features import (
    build_feature_packet,
    session_bounds,
    session_date_of,
    session_minute_bars,
    to_utc,
)
from regimepilot.gates import evaluate_gates
from regimepilot.models import (
    AccountHint,
    EvidencePacket,
    GatesEvidence,
    NewsEvidence,
    Observation,
    OhlcvBar,
    TradeAction,
    TradeProposal,
    UnderlyingEvidence,
    UNDERLYING_SYMBOL,
)
from regimepilot.regime import RegimeReading, classify_regime

# Backtest-only assumptions, absent from every live module. Documented here,
# in one place, so a reader of a scorecard knows exactly what to distrust.
ASSUMED_RISK_FREE_RATE = 0.04
ASSUMED_EXPIRATION_DAYS = 7
MIN_ASSUMED_SIGMA = 0.08
MIN_OPTION_PRICE = 0.01
CADENCE_MINUTES = 15
TREND_LOOKBACK_SESSIONS = 20
ADX_PERIOD = 14

# Regime labels in which momentum is treated as unreliable when
# ``regime_gate=True``: chop and drift are exactly the two regimes discussed
# as "trust this signal less" in the regime module's docstring.
LOW_TRUST_REGIMES = frozenset({"high_vol_chop", "low_vol_drift"})
REQUIRED_CONFIDENCE_IN_LOW_TRUST_REGIME = "high"

DecisionFn = Callable[[EvidencePacket], "object"]


class BacktestError(RuntimeError):
    """Raised for malformed input data. Never raised for a HOLD-only backtest."""


class SimulatedTrade(Observation):
    """One simulated entry and its simulated exit.

    Simplification #1 (see module docstring): ``exit_time`` is always the
    close of ``entry_time``'s session. A real position can live longer; this
    backtest does not simulate that yet.
    """

    entry_time: datetime
    exit_time: datetime
    option_type: str
    strike: float
    expiration_date: date
    entry_spot: float
    exit_spot: float
    entry_price: float
    exit_price: float
    qty: int
    pnl_usd: float
    regime_label: str
    confidence: str
    thesis: str


def load_minute_bars_csv(path: str) -> list[OhlcvBar]:
    """Read historical 1-minute OHLCV bars from a CSV.

    Expected columns: ``timestamp,open,high,low,close,volume``, one row per
    minute, ``timestamp`` in ISO 8601 (``2026-06-01T13:30:00+00:00`` or with a
    trailing ``Z``). This is the same shape Alpaca's historical bars endpoint
    returns when written to CSV, but this function never calls Alpaca: point
    it at a file exported however you like. Rows are returned sorted by
    timestamp; malformed rows raise ``BacktestError`` naming the row number,
    rather than silently dropping history a scorecard would then be wrong
    about.
    """
    bars: list[OhlcvBar] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            try:
                bars.append(
                    OhlcvBar(
                        timestamp=datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]) if row.get("volume") else None,
                    )
                )
            except (KeyError, ValueError) as error:
                raise BacktestError(f"{path}:{line_number}: {error}") from error
    bars.sort(key=lambda bar: bar.timestamp)
    return bars


def daily_bars_from_minute_bars(minute_bars: Sequence[OhlcvBar]) -> list[OhlcvBar]:
    """Aggregate 1-minute bars into one regular-session daily OHLC bar per session.

    Stamped 00:00 America/New_York on the session date, matching the
    convention ``features.previous_session_close`` and this project's tests
    already assume. A session with no priced bars is skipped rather than
    emitted as a bar of nulls, so it can never masquerade as a real prior
    close or as a real day of trend history.
    """
    by_session: dict[date, list[OhlcvBar]] = {}
    for bar in minute_bars:
        if bar.timestamp is None:
            continue
        by_session.setdefault(session_date_of(bar.timestamp), []).append(bar)

    daily: list[OhlcvBar] = []
    for session_day in sorted(by_session):
        session_bars = sorted(
            (b for b in by_session[session_day] if b.close is not None),
            key=lambda b: b.timestamp,
        )
        if not session_bars:
            continue
        opens_at, _ = session_bounds(session_day)
        highs = [b.high for b in session_bars if b.high is not None]
        lows = [b.low for b in session_bars if b.low is not None]
        daily.append(
            OhlcvBar(
                timestamp=opens_at.replace(hour=0, minute=0, second=0, microsecond=0),
                open=session_bars[0].open,
                high=max(highs) if highs else None,
                low=min(lows) if lows else None,
                close=session_bars[-1].close,
                volume=sum(b.volume for b in session_bars if b.volume is not None) or None,
            )
        )
    return daily


def _nearest_strike(spot: float, option_type: TradeAction | str) -> float:
    """The strike nearest ``spot``, rounded to the nearest whole dollar.

    Mirrors ``selector.py``'s "strike nearest the SPY midpoint" rule at a
    coarser grain: real SPY weekly strikes are $1 apart near the money, so
    rounding to the nearest dollar is the same answer the real chain would
    offer, without needing a real chain to ask.
    """
    return round(spot)


def _build_evidence(
    *,
    observed_at: datetime,
    minute_bars_so_far: Sequence[OhlcvBar],
    daily_bars_before_today: Sequence[OhlcvBar],
    session_close_at: datetime,
) -> tuple[EvidencePacket, RegimeReading]:
    """One cycle's worth of evidence, built the same way ``evidence.py`` builds it live."""
    packet = build_feature_packet(
        observed_at=observed_at,
        minute_bars=minute_bars_so_far,
        daily_bars=daily_bars_before_today,
        bid=None,
        ask=None,
        market_is_open=True,
        session_close_at=session_close_at,
    )
    gate_result = evaluate_gates(packet)
    regime = classify_regime(
        realized_vol_30m=packet.realized_vol_30m,
        return_60m=packet.return_60m,
        trend_bars=daily_bars_before_today[-TREND_LOOKBACK_SESSIONS:],
        current_iv=None,  # No historical option IV feed is available offline.
        iv_history=(),
        trend_period=ADX_PERIOD,
    )
    evidence = EvidencePacket(
        observed_at=observed_at,
        symbol=UNDERLYING_SYMBOL,
        gates=GatesEvidence(
            passed=gate_result.passed,
            hold_reason=gate_result.hold_reason,
            momentum_align=gate_result.labels.momentum_align,
        ),
        underlying=UnderlyingEvidence(
            market_is_open=packet.market_is_open,
            data_feed=packet.data_feed,
            minutes_since_open=packet.minutes_since_open,
            minutes_to_close=packet.minutes_to_close,
            spread_bps=packet.spread_bps,
            bar_age_seconds=packet.bar_age_seconds,
            return_15m=packet.return_15m,
            return_60m=packet.return_60m,
            return_since_open=packet.return_since_open,
            overnight_gap_pct=packet.overnight_gap_pct,
            realized_vol_30m=packet.realized_vol_30m,
        ),
        news=NewsEvidence(),
        account=AccountHint(),
        portfolio=None,
    )
    return evidence, regime


def _simulate_entry_and_exit(
    *,
    action: TradeAction,
    entry_time: datetime,
    entry_spot: float,
    exit_time: datetime,
    exit_spot: float,
    regime: RegimeReading,
    confidence: str,
    thesis: str,
) -> SimulatedTrade:
    option_type = "call" if action == "BUY_CALL" else "put"
    strike = _nearest_strike(entry_spot, option_type)
    expiration = entry_time.date() + timedelta(days=ASSUMED_EXPIRATION_DAYS)
    sigma = max(regime.realized_vol_annualized or MIN_ASSUMED_SIGMA, MIN_ASSUMED_SIGMA)

    entry_time_years = max((expiration - entry_time.date()).days, 1) / 365.0
    entry_price = max(
        black_scholes.option_price(
            option_type, entry_spot, strike, entry_time_years, ASSUMED_RISK_FREE_RATE, sigma
        ),
        MIN_OPTION_PRICE,
    )

    exit_time_years = max((expiration - exit_time.date()).days, 0) / 365.0
    exit_price = max(
        black_scholes.option_price(
            option_type, exit_spot, strike, exit_time_years, ASSUMED_RISK_FREE_RATE, sigma
        ),
        MIN_OPTION_PRICE,
    )

    qty = 1
    pnl_usd = round((exit_price - entry_price) * 100 * qty, 2)

    return SimulatedTrade(
        entry_time=entry_time,
        exit_time=exit_time,
        option_type=option_type,
        strike=strike,
        expiration_date=expiration,
        entry_spot=entry_spot,
        exit_spot=exit_spot,
        entry_price=round(entry_price, 4),
        exit_price=round(exit_price, 4),
        qty=qty,
        pnl_usd=pnl_usd,
        regime_label=regime.label,
        confidence=confidence,
        thesis=thesis,
    )


def apply_regime_gate(proposal: TradeProposal, regime: RegimeReading) -> TradeProposal:
    """Force ``HOLD`` on a non-HOLD proposal made in a low-trust regime at low/medium confidence.

    This is the backtest-only implementation of the override described in
    ``regime.py``'s module docstring: momentum in ``high_vol_chop`` or
    ``low_vol_drift`` is treated as unreliable, so only a ``"high"``-confidence
    proposal is allowed through unchanged. A ``HOLD`` proposal is always
    passed through as-is -- there is nothing to gate.

    Deliberately a plain function taking and returning a ``TradeProposal``,
    not a parameter baked into ``decision.stub_proposal`` itself: this keeps
    the override testable and comparable on its own, and keeps open the
    ability to A/B it against the ungated backtest by simply toggling
    ``run_backtest(..., regime_gate=True/False)`` rather than needing two
    copies of the decision logic.
    """
    if proposal.action == "HOLD":
        return proposal
    if regime.label not in LOW_TRUST_REGIMES:
        return proposal
    if proposal.confidence == REQUIRED_CONFIDENCE_IN_LOW_TRUST_REGIME:
        return proposal

    return TradeProposal(
        observed_at=proposal.observed_at,
        symbol=proposal.symbol,
        action="HOLD",
        confidence=proposal.confidence,
        thesis=(
            f"Regime override: {regime.label} treats momentum as unreliable; "
            f"original proposal was {proposal.action} at {proposal.confidence} "
            f"confidence, below the required {REQUIRED_CONFIDENCE_IN_LOW_TRUST_REGIME}. "
            f"Original thesis: {proposal.thesis}"
        ),
        evidence_used=proposal.evidence_used,
        gate_skipped=proposal.gate_skipped,
        model=proposal.model,
    )


def run_backtest(
    minute_bars: Sequence[OhlcvBar],
    *,
    decision_fn: DecisionFn | None = None,
    cadence_minutes: int = CADENCE_MINUTES,
    regime_gate: bool = False,
) -> list[SimulatedTrade]:
    """Replay ``minute_bars`` cycle by cycle and return every simulated trade.

    ``decision_fn`` defaults to ``decision.stub_proposal`` -- the rule-based
    momentum stand-in -- so a backtest run needs no OpenRouter key and is
    fully reproducible. Pass ``decision.propose_trade`` (or a wrapper around
    it) to backtest the LLM path instead; be aware that then re-running the
    same history can produce different trades, since the model is not
    deterministic, which is exactly why the stub is the default for a
    repeatable scorecard.

    ``regime_gate``, when ``True``, runs every non-HOLD proposal through
    ``apply_regime_gate`` before it can open a position -- see that
    function's docstring. Defaults to ``False`` so existing callers and
    existing scorecards are unaffected; run the same history with
    ``regime_gate=False`` and ``regime_gate=True`` and diff the two
    scorecards from ``score.py`` to see whether gating actually helps before
    ever touching the live decision path in ``evidence.py``/``decision.py``.

    One cycle is evaluated every ``cadence_minutes`` inside each session
    (mirroring the live runner's 15-minute loop); at most one simulated
    position is open at a time, opened only when none is open and the
    proposal is not ``HOLD``, and always closed at that session's last bar
    (Simplification #1 in the module docstring).

    ``decision.py`` is imported here, not at module level, so that importing
    ``backtest`` alone -- to just simulate fills against an already-computed
    proposal, say -- never pulls in ``evidence.py``'s account/Alpaca SDK
    import chain. Mirrors the same convention ``selector.main`` already uses.
    """
    from regimepilot.decision import build_gate_hold_proposal, stub_proposal

    if decision_fn is None:
        decision_fn = stub_proposal

    daily_bars = daily_bars_from_minute_bars(minute_bars)
    by_session: dict[date, list[OhlcvBar]] = {}
    for bar in minute_bars:
        if bar.timestamp is not None and bar.close is not None:
            by_session.setdefault(session_date_of(bar.timestamp), []).append(bar)

    trades: list[SimulatedTrade] = []
    open_entry: dict | None = None  # holds action/time/spot/regime/confidence/thesis, or None

    for session_day in sorted(by_session):
        session_bars = sorted(by_session[session_day], key=lambda b: b.timestamp)
        _, session_close_at = session_bounds(session_day)
        daily_bars_before_today = [b for b in daily_bars if b.timestamp.date() < session_day]

        cursor = session_bars[0].timestamp
        while cursor <= session_bars[-1].timestamp:
            bars_so_far = [b for b in session_bars if b.timestamp <= cursor]
            if bars_so_far:
                evidence, regime = _build_evidence(
                    observed_at=cursor + timedelta(minutes=1),
                    minute_bars_so_far=bars_so_far,
                    daily_bars_before_today=daily_bars_before_today,
                    session_close_at=session_close_at,
                )
                if evidence.gates.passed:
                    proposal = decision_fn(evidence)
                else:
                    proposal = build_gate_hold_proposal(evidence)

                if regime_gate:
                    proposal = apply_regime_gate(proposal, regime)

                if open_entry is None and proposal.action in ("BUY_CALL", "BUY_PUT"):
                    open_entry = {
                        "action": proposal.action,
                        "time": cursor,
                        "spot": bars_so_far[-1].close,
                        "regime": regime,
                        "confidence": proposal.confidence,
                        "thesis": proposal.thesis,
                    }
            cursor += timedelta(minutes=cadence_minutes)

        if open_entry is not None:
            exit_spot = session_bars[-1].close
            trades.append(
                _simulate_entry_and_exit(
                    action=open_entry["action"],
                    entry_time=open_entry["time"],
                    entry_spot=open_entry["spot"],
                    exit_time=session_bars[-1].timestamp,
                    exit_spot=exit_spot,
                    regime=open_entry["regime"],
                    confidence=open_entry["confidence"],
                    thesis=open_entry["thesis"],
                )
            )
            open_entry = None

    return trades


def main(argv: Sequence[str] | None = None) -> int:
    """Run a backtest from a CSV of historical minute bars and print (or emit JSON) the trades.

    Usage: ``python -m regimepilot.backtest --csv path/to/bars.csv [--json] [--regime-gate]``
    Never touches the network or a credential; ``--csv`` is required.
    """
    tolerant_console()
    arguments = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in arguments
    regime_gate = "--regime-gate" in arguments
    arguments = [a for a in arguments if a not in ("--json", "--regime-gate")]

    if "--csv" not in arguments or arguments.index("--csv") + 1 >= len(arguments):
        print(
            "usage: python -m regimepilot.backtest --csv path/to/bars.csv [--json] [--regime-gate]",
            file=sys.stderr,
        )
        return 1
    csv_path = arguments[arguments.index("--csv") + 1]

    try:
        bars = load_minute_bars_csv(csv_path)
        trades = run_backtest(bars, regime_gate=regime_gate)
    except BacktestError as error:
        print(f"backtest error: {error}", file=sys.stderr)
        return 1

    if as_json:
        import json

        print(json.dumps([t.model_dump(mode="json") for t in trades], indent=2))
    else:
        print(f"{len(trades)} simulated trade(s) from {csv_path}")
        for t in trades:
            print(
                f"  {t.entry_time.isoformat()} {t.option_type:<4} strike={t.strike:<7} "
                f"regime={t.regime_label:<14} pnl=${t.pnl_usd:+.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
