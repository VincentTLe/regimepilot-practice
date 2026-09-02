# Trading review — first live day (2026-09-01)

Written 2026-09-02 by Claude acting as the system's momentum-trader decision
layer, after running 32 live cycles on 2026-09-01 (13:28–16:01 ET) via
`/paca-agent`. Evidence: `logs/cycles.jsonl` (106 live cycles across 8/31–9/1),
the code as of commit `423a785`, and the closed-trade P&L. The changes this
review recommended were approved and landed in commit `cbe1cff`.

## Verdict

The architecture is right: deterministic screening, sizing, and exits with a
discretionary decision layer on top, an append-only journal, and risk caps that
cannot be bypassed. The safety layer visibly earned its keep on day one (the
pre-order recheck vetoed a QQQ put whose quotes had widened; the trade would
have lost). The losses came from three specific, fixable places — signal noise,
spread structure, and a mis-sized screener floor — not from the design.

## What the day's data showed

- **45 entry attempts → 6 orders.** 36 died as `no_spread`, 3 at the pre-order
  recheck, 1 at risk caps. The screener, not the signal, was the bottleneck.
- **SPY alone: 925 `too_narrow` rejections.** `min_width_pct: 0.02` = a $15.22
  width floor on a $761 underlying — the most liquid option chain in the world
  was nearly untradeable.
- **All 4 closed round trips were reversal-exit losses** (−$18, −$123, −$133,
  −$25 ≈ −$298 total), held 52 min–2 h 46 min. Stop-loss and take-profit never
  fired once — the reversal exit always cut first.
- **The MACD histogram flips sign 3–6×/day/symbol on 5m bars**, and a cross
  fired on *any* sign flip (even ±0.001). Each flip was both an entry candidate
  and a reversal-exit trigger: enter on a wiggle, exit on the next wiggle,
  pay 4 legs of option friction per round trip.
- **The spread ranker picked lottery tickets.** Max reward-to-risk ranking
  mathematically favors deep OTM: it chose a TSLA 380/390 call spread on a
  $356 stock — debit 11% of width, roughly a 0.15-delta bet.

## Findings and what was done (commit `cbe1cff`)

### 1. MACD zero-cross had no magnitude threshold — the dominant loss source

A momentum event should represent impulse. The 2-ATR gap/breakout events do;
a bare histogram sign flip does not. **Fix:** `macd_cross_*` now fires only
when `|histogram| ≥ macd_min_hist_atr × ATR` (shipped 0.05). Reversal exits
consume the same events, so whipsaw exits inherit the fix automatically.

Calibration against 9/1 data (|hist|/ATR at the moment of cross): the good
signals pass — TSLA +0.107 (0.16), AAPL +0.031 (0.054) — while every trade the
human decider passed on as noise is now blocked mechanically: MSFT +0.0007
(0.001), TSLA −0.0014 (0.002), SPY ±0.012–0.0135 (0.025–0.03).

### 2. RSI was computed but never used — exhaustion filtering lived in prompt text

Every RSI judgment ("don't chase a gap at RSI 75") depended on the decision
layer remembering to make it. **Fix:** a deterministic entry gate
(`rsi_exhausted`): CALL events are dropped at RSI ≥ `rsi_overbought` (70), PUT
events at RSI ≤ `rsi_oversold` (30). Entries only — the exit path still sees
raw events, so a capitulation gap at RSI 25 still closes a held call spread.

### 3. Reward-to-risk ranking bought deep-OTM lottery spreads

A momentum spread trader buys the long leg near ATM (delta ~0.4–0.55), paying
25–45% of the width, so the position responds to the move being traded.
**Fix:** the debit must sit in `[min_debit_frac, max_debit_frac] × width`
(shipped 0.25–0.45), enforced both in the screener and again at the pre-order
recheck. Ranking stays max reward-to-risk *within* that band.

### 4. Width floor mis-sized for index ETFs

`min_width_pct` 0.02 → 0.01. With the debit band in place, narrower spreads
remain economically sane, and SPY/QQQ can actually form candidates.

### 5. Small bug: flat-price bars scored RSI 100

`loss == 0` forced RSI to 100 even when `gain == 0` too. A flat run is neutral:
now 50. (Mattered for thin symbols on the IEX feed.)

## Deliberately unchanged

- **`bar_timeframe: 5m`** — 15m bars would give better signals but near-zero
  trades in the hackathon's remaining two days. Revisit after Sep 4.
- **Risk fractions** (0.5% per entry / 1.5% per underlying / 10% total) —
  conservative and appropriate; sizing was never the problem.
- **Exit levels** (stop −50%, take-profit +100%, exit at DTE ≤ 2) — they never
  got a chance to act behind the hair-trigger reversal exit; judge them now
  that reversals require a real signal.
- **Liquidity filters** (OI ≥ 100, quote age ≤ 10 s, leg spread ≤ 350 bps) —
  they rejected a lot, but on the IEX feed they were doing genuine safety work.

## Deferred recommendations (tracked in TODO.md)

1. **Regular-hours session filter** — indicators are computed over extended-hours
   IEX bars. Thin overnight bars shrink ATR (the event denominator), and the
   09:30 gap event compares against the previous *extended-hours* bar, so real
   overnight gaps are largely invisible. Highest-value remaining item.
2. **Event cooldown / one-shot bookkeeping** — the same 5m bar persists across
   cycles (median cycle gap was 4.2 min), so one event can be acted on twice.
3. **Higher-timeframe trend alignment** — a 1h/daily EMA agreement gate so 5m
   longs only fire with the larger trend (the pre-rewrite journal had a
   `momentum_align` gate; it was lost in the rewrite).
4. **Exit-mark quote sanity** — stop/take-profit marks bypass `check_leg`, so a
   stale or absurdly wide option quote can trigger them.

## How to judge the changes

Watch the next live day's journal for: the mix of `rejected` reasons (if
`debit_out_of_band` dominates, loosen `min_debit_frac` toward 0.20); whether
SPY/QQQ now produce entries; holding times (should lengthen from ~1 h toward
multi-hour/overnight); and whether stop/take-profit exits finally fire. The
thresholds were calibrated on one session of data — treat them as a first
estimate, not truth.
