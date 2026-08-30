"""Turn a list of SimulatedTrade records into one comparable scorecard.

Pure module: takes ``backtest.SimulatedTrade`` objects (or real fills read
from ``logs/cycles.jsonl``, once wired up) and a baseline price series, and
computes the handful of numbers that answer "is this strategy worth
promoting": win rate, profit factor, a Sharpe-like risk-adjusted return, max
drawdown, and a buy-and-hold comparison. Nothing here trades, backtests, or
touches the network.

Naming is deliberately conservative: ``sharpe_per_trade`` and
``sharpe_annualized_approx`` are named to say what they actually are, not to
borrow the authority of a textbook Sharpe ratio computed on evenly spaced
daily returns, which this project's trade-triggered (not calendar-triggered)
entries do not produce.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Sequence
from datetime import date

from regimepilot.backtest import BacktestError, SimulatedTrade, load_minute_bars_csv, run_backtest
from regimepilot.console import tolerant_console
from regimepilot.models import Observation, OhlcvBar

# The methodology's own per-trade premium cap (README, "Approved methodology
# 2026-08-27"), used only to express drawdown as a percentage of a meaningful
# base rather than an arbitrary one.
REFERENCE_ACCOUNT_EQUITY_USD = 100_000.0


class BacktestScorecard(Observation):
    """One scorecard for one list of simulated (or real) trades.

    Every ratio field is ``None`` when there is not enough data to compute it
    honestly (fewer than two trades for a standard deviation, no losing
    trades to divide by for profit factor) rather than a placeholder zero,
    following the same null policy ``features.py`` and ``regime.py`` use.
    """

    trade_count: int
    win_rate: float | None = None
    total_pnl_usd: float = 0.0
    average_pnl_usd: float | None = None
    profit_factor: float | None = None
    sharpe_per_trade: float | None = None
    sharpe_annualized_approx: float | None = None
    max_drawdown_usd: float = 0.0
    max_drawdown_pct_of_reference_equity: float | None = None
    baseline_buy_and_hold_return_pct: float | None = None


def _per_trade_return(trade: SimulatedTrade) -> float:
    """Return on the premium paid for one trade: pnl divided by cost basis."""
    cost_basis = trade.entry_price * 100 * trade.qty
    if cost_basis <= 0:
        return 0.0
    return trade.pnl_usd / cost_basis


def _max_drawdown(pnls_in_time_order: Sequence[float]) -> float:
    """Largest peak-to-trough drop in the cumulative-PnL equity curve, in dollars."""
    cumulative = 0.0
    peak = 0.0
    worst = 0.0
    for pnl in pnls_in_time_order:
        cumulative += pnl
        peak = max(peak, cumulative)
        worst = min(worst, cumulative - peak)
    return abs(worst)


def _buy_and_hold_return_pct(daily_bars: Sequence[OhlcvBar], start: date, end: date) -> float | None:
    """SPY's own return, close to close, over ``[start, end]``. ``None`` if unavailable."""
    in_range = sorted(
        (b for b in daily_bars if b.close is not None and start <= b.timestamp.date() <= end),
        key=lambda b: b.timestamp,
    )
    if len(in_range) < 2:
        return None
    return (in_range[-1].close / in_range[0].close - 1) * 100


def compute_scorecard(
    trades: Sequence[SimulatedTrade], daily_bars: Sequence[OhlcvBar] = ()
) -> BacktestScorecard:
    """Score a completed list of trades. Empty input scores as zero trades, not an error."""
    if not trades:
        return BacktestScorecard(trade_count=0)

    ordered = sorted(trades, key=lambda t: t.exit_time)
    pnls = [t.pnl_usd for t in ordered]
    returns = [_per_trade_return(t) for t in ordered]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / len(pnls)
    total_pnl = sum(pnls)
    average_pnl = total_pnl / len(pnls)

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    sharpe_per_trade = None
    sharpe_annualized = None
    if len(returns) >= 2:
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_return = math.sqrt(variance)
        if std_return > 0:
            sharpe_per_trade = mean_return / std_return
            span_days = (ordered[-1].exit_time - ordered[0].entry_time).days
            if span_days > 0:
                trades_per_year = len(ordered) / (span_days / 365.25)
                sharpe_annualized = sharpe_per_trade * math.sqrt(trades_per_year)

    max_dd = _max_drawdown(pnls)
    max_dd_pct = (max_dd / REFERENCE_ACCOUNT_EQUITY_USD) * 100 if REFERENCE_ACCOUNT_EQUITY_USD > 0 else None

    baseline = None
    if daily_bars:
        baseline = _buy_and_hold_return_pct(
            daily_bars, ordered[0].entry_time.date(), ordered[-1].exit_time.date()
        )

    return BacktestScorecard(
        trade_count=len(ordered),
        win_rate=win_rate,
        total_pnl_usd=round(total_pnl, 2),
        average_pnl_usd=round(average_pnl, 2),
        profit_factor=round(profit_factor, 3) if profit_factor is not None else None,
        sharpe_per_trade=round(sharpe_per_trade, 4) if sharpe_per_trade is not None else None,
        sharpe_annualized_approx=round(sharpe_annualized, 3) if sharpe_annualized is not None else None,
        max_drawdown_usd=round(max_dd, 2),
        max_drawdown_pct_of_reference_equity=round(max_dd_pct, 3) if max_dd_pct is not None else None,
        baseline_buy_and_hold_return_pct=round(baseline, 3) if baseline is not None else None,
    )


def compute_scorecards_by_regime(
    trades: Sequence[SimulatedTrade], daily_bars: Sequence[OhlcvBar] = ()
) -> dict[str, BacktestScorecard]:
    """Split ``trades`` by ``regime_label`` and score each group separately.

    Answers "did the current decision logic actually perform differently
    across regimes" -- the question regime classification exists to raise,
    even though (as of this patch) nothing yet gates entries on the label.
    Each group's baseline uses the same ``daily_bars``, so every regime's
    scorecard is still comparable against the same buy-and-hold reference,
    not a regime-local one.

    A label that never occurred in ``trades`` is simply absent from the
    result -- there is no zero-trade entry synthesized for it, matching this
    project's convention of never fabricating a data point that was not
    observed.
    """
    by_label: dict[str, list[SimulatedTrade]] = {}
    for t in trades:
        by_label.setdefault(t.regime_label, []).append(t)
    return {label: compute_scorecard(group, daily_bars) for label, group in by_label.items()}


def format_summary(card: BacktestScorecard) -> str:
    """A compact, honest report, matching this project's other format_summary functions."""

    def fmt(value, digits: int = 2, suffix: str = "") -> str:
        return "null" if value is None else f"{value:.{digits}f}{suffix}"

    return "\n".join(
        [
            f"RegimePilot backtest scorecard  trades={card.trade_count}",
            f"  {'win_rate':<32} {fmt(card.win_rate, 3)}",
            f"  {'total_pnl_usd':<32} {fmt(card.total_pnl_usd)}",
            f"  {'average_pnl_usd':<32} {fmt(card.average_pnl_usd)}",
            f"  {'profit_factor':<32} {fmt(card.profit_factor, 3)}",
            f"  {'sharpe_per_trade':<32} {fmt(card.sharpe_per_trade, 4)}",
            f"  {'sharpe_annualized_approx':<32} {fmt(card.sharpe_annualized_approx, 3)}",
            f"  {'max_drawdown_usd':<32} {fmt(card.max_drawdown_usd)}",
            f"  {'max_drawdown_pct_of_ref_equity':<32} {fmt(card.max_drawdown_pct_of_reference_equity, 3, '%')}",
            f"  {'baseline_buy_and_hold_return':<32} {fmt(card.baseline_buy_and_hold_return_pct, 3, '%')}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run a backtest from a CSV of historical minute bars and print its scorecard.

    Usage: ``python -m regimepilot.score --csv path/to/bars.csv [--json] [--regime-gate]``
    """
    tolerant_console()
    arguments = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in arguments
    regime_gate = "--regime-gate" in arguments
    arguments = [a for a in arguments if a not in ("--json", "--regime-gate")]

    if "--csv" not in arguments or arguments.index("--csv") + 1 >= len(arguments):
        print(
            "usage: python -m regimepilot.score --csv path/to/bars.csv [--json] [--regime-gate]",
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

    from regimepilot.backtest import daily_bars_from_minute_bars

    daily_bars = daily_bars_from_minute_bars(bars)
    card = compute_scorecard(trades, daily_bars)
    by_regime = compute_scorecards_by_regime(trades, daily_bars)

    if as_json:
        print(
            json.dumps(
                {
                    "overall": card.model_dump(mode="json"),
                    "by_regime": {label: c.model_dump(mode="json") for label, c in by_regime.items()},
                },
                indent=2,
            )
        )
    else:
        print(format_summary(card))
        if by_regime:
            print("\nBy regime label:")
            for label in sorted(by_regime):
                print(f"\n[{label}]")
                print(format_summary(by_regime[label]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
