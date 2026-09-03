# Letting the agent test its own rules

*tape/paca branch — proposal for the team, 2026-09-03*

## The problem

PACA's entries are the output of a stack of hand-written rules: a bar event
(gap / breakout / MACD cross) or a tape event, an RSI exhaustion filter, a tape
agreement gate, the LLM decider, the option screener's liquidity gates, the
risk caps. Every rule was set once, from a backtest, and then never questioned
again. On 2026-09-03 the stack sat through MSFT +4%, TSLA +4.5% and SNOW +21%
with no trade, because the RSI filter blocked every gap-open candidate and the
event set is blind to steady trends. Nobody could see that from the P&L: a rule
that blocks a winner leaves no trace.

Tan's question was: can the LLM sometimes break a rule on purpose, and can we
mark those trades and follow them, the way training a network sometimes takes a
random step to find out whether the current rule is really the best one? That
mechanism has a name in reinforcement learning: **exploration** (ε-greedy: with
probability ε take an action other than the one the policy says is best, so the
policy keeps learning). In production systems it is called a
**champion / challenger** or A/B test. Dropout is a different trick
(regularisation while training), but the intuition is the same: without a
controlled dose of rule-breaking, a rule is never re-examined.

## Two ways to do it, in order

### 1. Counterfactual replay — grade every decision, not just the trades (built)

The loop already journals, every five minutes, **every** candidate it looked at:
price, tape imbalance and print count, RSI, EMA anchors, the raw events with
their direction *before* the RSI and tape filters, the rule that blocked it, the
entries it attempted, and since today every decider pass with the model's stated
reason. That is a complete record of the roads not taken.

`review_rules.py` refetches the session's 5-minute bars and measures what the
underlying did after each decision — 60 minutes later and at the session close,
in ATR units signed by the candidate's direction — for every group:

| group | meaning |
|---|---|
| `entered` | an order was submitted |
| `no_spread` | the decider chose it, the option screener found nothing tradeable |
| `llm_pass` | offered to the decider, it declined (thesis journaled) |
| `flow_disagree`, `rsi_exhausted`, ... | blocked by that rule, graded on its raw events |
| `tape_only` | every candidate with \|flow\| ≥ 0.10 graded on the flow sign alone, split by trend alignment |

A rule earns its place when the candidates it blocks do worse than the ones it
lets through. A rule whose blocked candidates keep winning is a rule to relax.
This costs nothing: no random trades, tens of graded decisions per day instead
of one, and the same yardstick as the 90-session backtests (the underlying's
move; option P&L is not modelled because the account has no option history).

First run, 2026-09-03 up to 13:00 ET (small sample, one day):

```
tape_only, flow WITH the trend:    0.10-0.25 -> +0.39 ATR to the close (n=94), 0.25-0.40 -> +0.66 (n=21)
tape_only, flow AGAINST the trend: 0.10-0.25 -> -1.20 (n=150),             0.25-0.40 -> -1.37 (n=36)
llm_pass: GLD PUT declined at 10:35 ET went +1.5 ATR
no_spread: the three chosen-but-untradeable names went -1.15 on average (the screener got lucky)
```

Same picture as the 90-session backtest: the tape is the edge, but only in the
direction of the trend anchors.

### 2. Controlled exploration — real trades that break one rule (planned)

With probability ε per decision point (say 15%) and at most N times a day, the
loop picks one blocked candidate, switches off **one** rule for it (the tape
gate, the RSI filter, the event requirement, the spread cap, ...), hands it to
the decider flagged `explore`, and trades it at half size. The journal records
which rule was broken; `review_rules.py` compares the explore trades with the
rule-following ones over time. A rule that keeps losing to its own exceptions
gets changed.

Only two constraints are not up for exploration, because they are about the
account rather than the strategy: **paper trading only** and the **position
size caps**. Everything else is a hypothesis.

Honest statistics: at 2–5 trades a day, one explore trade a day needs weeks
before the comparison means anything. Method 1 reaches the same conclusions
faster because it grades every candidate, taken or not. Method 2 is worth
having because a blocked candidate can only be graded on the underlying, while
an explore trade shows the whole thing — fill, option friction, exit — and
because an agent that runs controlled experiments on its own rules is a story
worth telling.

## How to run it

```bash
uv run --env-file .env review_rules.py --date 2026-09-03 --html surge_artifacts/paca-backtest/review_2026-09-03.html --deploy
```

The page lands next to the walk-forward report on the backtest site.
