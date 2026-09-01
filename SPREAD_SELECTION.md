# Spread selection & ranking

How one debit vertical spread is chosen per entry. All code lives in
`options_screener.py` (pure functions); every threshold is a key in
`settings.yaml` (screener section) — the numbers below are the shipped
defaults. Rules approved 2026-08-31, width/moneyness/ranking revised
2026-09-01.

## Pipeline

The screener runs once per entry attempt, on the single `(symbol, direction)`
the decision layer picked:

1. **Expiry** — nearest listed expiry (weeklies included) at least
   `min_dte` (5) days out, ignoring anything past
   `max_expiry_lookahead_days` (45). `pick_expiration`
2. **Strike universe** — strikes within ±`strike_band_pct` (10%) of spot.
   With `otm_only: true`: out-of-the-money strikes only (calls above spot,
   puts below), **plus the one at-the-money strike bracketing spot** on the
   ITM side (calls: highest strike ≤ spot; puts: lowest ≥ spot) — typically
   the tightest-quoted strike in the chain. `enumerate_spreads`
3. **Per-leg filter** — each strike's contract must have:
   open interest ≥ `min_open_interest` (100); a two-sided quote with
   `bid ≤ ask`, both positive; a timestamp within `max_quote_age_seconds`
   (10 s) of Alpaca's server clock *in either direction* (snapshots are
   fetched after the clock read, so fresh quotes may postdate it slightly);
   bid-ask spread ≤ `max_leg_spread_bps` (500) of the mid; implied
   volatility present. `check_leg`
4. **Pairing** — every surviving strike pair whose width falls between
   `min_width_pct` (2%) and `max_width_pct` (5%) of spot. Bull call: long
   the lower strike, short the higher; bear put reversed.
5. **Debit sanity** — the marketable debit `ask(long) − bid(short)` must
   satisfy `min_net_debit` (0.05) ≤ debit < width.
6. **Ranking** — see below; the top-ranked spread is the pick.

Every rejection in steps 2–5 is tallied and journaled per cycle in
`logs/cycles.jsonl` under `entry.screen_rejections` (e.g. `low_open_interest`,
`wide_spread`, `too_narrow`, `bad_debit`) — the first place to look when a
cycle reports `no_spread`.

The same `check_leg` filter runs a second time immediately before submission
(the pre-submit re-check in `cli._attempt_entry`): both legs are re-quoted
against a fresh clock, and the debit is re-sized, so a spread that decayed
between screening and submission is refused rather than sent.

## What each filter protects against

Every knob exists to keep a specific bad trade out. Journal labels in
parentheses.

- **`min_dte` (5) / `max_expiry_lookahead_days` (45)** (`no_expiration`) —
  the floor keeps the spread out of the final-week gamma/theta zone, where a
  small adverse move destroys the debit before the momentum thesis can play
  out (a separate exit rule closes anything that reaches DTE ≤ 2); the cap
  stops the screener from drifting into far-dated expiries whose premium is
  mostly time value unrelated to a bar-scale momentum signal.
- **`strike_band_pct` (0.10)** (`too_few_strikes_in_band`) — bounds both the
  API fetch and the universe: strikes beyond ±10% of spot are either deep ITM
  (all intrinsic, wide quotes) or lottery tickets, and neither belongs in a
  3–5%-wide vertical. It is applied twice: in the Alpaca contracts request
  (`broker.fetch_contracts`) and again locally.
- **`otm_only` (true)** — ITM legs price mostly intrinsic value, so they
  inflate the debit without adding payoff leverage, and they usually quote
  wide. The one **ATM bracketing strike** (calls: highest ≤ spot; puts:
  lowest ≥ spot) is deliberately kept because it is typically the
  tightest-quoted, highest-OI strike in the chain — observed repeatedly on
  AAPL, where the just-ITM strike quoted ~34–320 bps while its OTM neighbors
  sat at 400–700.
- **`min_width_pct` (2%) / `max_width_pct` (5%)** (`too_narrow` / `too_wide`)
  — the floor keeps the max payoff large enough to matter after slippage and
  the stop/TP exits; the cap bounds risk per spread and also limits how far
  the reward-to-risk ranking can chase cheap, low-probability wide spreads.
- **`min_open_interest` (100, per leg)** (`low_open_interest`) — open
  interest is the proxy for "someone will be on the other side when the exit
  order goes out". Entries are marketable limits, but every spread must also
  be *closed*; a leg with no open interest can strand the position. In
  practice this is the dominant rejection on single-name chains (AAPL band
  strikes routinely tally 5–20), which is expected — most of a chain is dead.
  Raising it much above ~500 effectively restricts trading to index products
  (SPY/QQQ/IWM).
- **`max_quote_age_seconds` (10, both directions)** (`stale_quote` /
  `future_quote`) — a quote older than 10 s vs Alpaca's server clock may not
  reflect the market the order will meet. The same tolerance applies in the
  *future* direction because snapshots are fetched after the clock read, so
  the freshest quotes legitimately postdate it by the fetch latency
  (2026-09-01 fix); only a timestamp more than 10 s ahead — genuine clock
  garbage — is rejected.
- **`max_leg_spread_bps` (500)** (`wide_spread`) — bounds slippage: the plan
  prices the entry at `ask(long) − bid(short)`, so wide legs both worsen the
  price and make the mid-based mark used by the stop/TP exits unreliable. In
  practice this is the *binding* filter: observed ATM option legs oscillate
  roughly 200–700 bps intraday, so small changes to this cap decide whether a
  cycle finds any spread at all. Too tight → `wide_spread` dominates the
  tallies and every cycle ends `no_spread`; too loose → fills land far from
  the screened debit and stop/TP levels drift from reality.
- **Two-sided quote sanity** (`no_quote` / `crossed_quote` / `missing_iv`) —
  data-quality guards, not tunables: a missing side, a non-positive price, a
  bid above the ask (typically a zero-bid far-OTM strike), or absent implied
  volatility all mean the feed cannot support a sane price for that leg.
- **`min_net_debit` (0.05)** (`bad_debit`, shared with `debit ≥ width`) —
  floors out junk spreads whose entire debit is inside quote noise; the
  upper sanity bound `debit < width` rejects pairs whose quotes imply a
  guaranteed loss at expiry.

## Ranking rule (current)

`rank_spreads` sorts by **reward-to-risk, highest first**:

```
reward_to_risk = (width − net_debit) / net_debit
```

That is the payoff multiple if the spread finishes fully in the money: risk
the debit, collect `width − debit`. Ties go to the **tighter combined leg
quotes** (summed bid-ask bps of both legs), i.e. the more fillable spread.

Rationale: entries fire on momentum events, so the bet is directional — pay
as little as possible per dollar of maximum payout, and among equals prefer
the spread whose quotes suggest a fill near mid.

Known bias worth watching: reward-to-risk favors the widest allowed spread
with the furthest-OTM short leg (cheap debit, but lower probability of
reaching max payout). The width band bounds how far this can stretch.

## Alternatives considered (not implemented)

Each of these is a one-line change to the sort key in `rank_spreads`
(plus tests). Switching is a **methodology change** per CLAUDE.md — decide
first, then implement.

| Rule | Sorts by | Pro | Con |
|---|---|---|---|
| Flattest IV skew *(the rule before 2026-09-01)* | `abs(IV_short − IV_long)` ascending, ties → higher combined OI | Avoids overpaying for the long leg's volatility | Ignores payoff shape entirely |
| Probability-weighted EV | `P·(width − debit) − (1 − P)·debit`, P ≈ short-strike delta | Balances payoff against likelihood — the most principled | Needs greeks plumbed from Alpaca snapshots into `LegQuote`; delta is only a proxy for P(max payoff) |
| Debit-to-width target band | Prefer `debit / width` in ~0.25–0.40 | Classic vertical heuristic; balances probability vs payoff without greeks | A band, not a total order — still needs a tiebreak inside the band |
| Highest delta / ATM-first | Long strike closest to spot | Maximizes the chance the momentum move pays at all | Most expensive per dollar of payout (lowest reward-to-risk) |
| Execution quality | Combined leg bid-ask bps ascending | Best fills, least slippage | Ignores payoff; already serves as the current tiebreak |
| Composite score | Weighted blend (e.g. normalized reward-to-risk + liquidity) | Most expressive | Most knobs to tune and justify — against the keep-it-small rule unless clearly needed |

The pragmatic middle ground, if reward-to-risk proves too aggressive in
paper trading: the debit-to-width band, or probability-weighted EV once
greeks are worth the plumbing.
