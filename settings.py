"""Loads and validates settings.yaml — every trader-tunable knob in one place.

The file is validated key by key: a missing key, an unknown (typo'd) key, a
wrong type, or an out-of-range value raises SettingsError naming the exact key.
Validated values are exposed as module constants (settings.STOP_FRACTION etc.),
loaded once at import so any command fails fast on a bad file.

SDK-free on purpose: the pure modules (signals, positions, options_screener)
import this without pulling in the Alpaca SDK.
"""

from __future__ import annotations

from pathlib import Path

import yaml

SETTINGS_PATH = Path(__file__).parent / "settings.yaml"

_TIMEFRAME_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}


class SettingsError(Exception):
    pass


def parse_timeframe(raw: object) -> tuple[int, str, int]:
    """'15m' -> (15, 'm', 900). Raises SettingsError on anything unrecognized."""
    text = str(raw).strip().lower()
    if len(text) < 2 or text[-1] not in _TIMEFRAME_SECONDS or not text[:-1].isdigit():
        raise SettingsError(
            f"settings.yaml: bar_timeframe must look like 5m/15m/1h/1d/1w, got {raw!r}"
        )
    amount = int(text[:-1])
    if amount < 1:
        raise SettingsError("settings.yaml: bar_timeframe amount must be at least 1")
    return amount, text[-1], amount * _TIMEFRAME_SECONDS[text[-1]]


def _fail(path: str, message: str, value: object) -> None:
    raise SettingsError(f"settings.yaml: {path}: {message}, got {value!r}")


def _section(raw: dict, name: str, keys: set[str]) -> dict:
    section = raw.get(name)
    if not isinstance(section, dict):
        _fail(name, "must be a mapping section", section)
    unknown = set(section) - keys
    if unknown:
        _fail(name, f"unknown key(s) {sorted(unknown)} — a typo?", sorted(unknown))
    missing = keys - set(section)
    if missing:
        _fail(name, f"missing required key(s) {sorted(missing)}", None)
    return section


def _number(section: dict, path: str, key: str, lo: float, hi: float = float("inf"),
            *, lo_open: bool = False, hi_open: bool = False) -> float:
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{path}.{key}", "must be a number", value)
    below = value <= lo if lo_open else value < lo
    above = value >= hi if hi_open else value > hi
    if below or above:
        bounds = f"{'(' if lo_open else '['}{lo}, {hi}{')' if hi_open else ']'}"
        _fail(f"{path}.{key}", f"must be in {bounds}", value)
    return float(value)


def _integer(section: dict, path: str, key: str, lo: int) -> int:
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{path}.{key}", "must be a whole number", value)
    if value < lo:
        _fail(f"{path}.{key}", f"must be at least {lo}", value)
    return value


def _string_list(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(s, str) or not s.strip() for s in value):
        _fail(path, "must be a list of symbols/strings", value)
    return tuple(dict.fromkeys(s.strip().upper() for s in value))


_TOP_KEYS = {"symbols", "bar_timeframe", "loop_interval_seconds",
             "signals", "scanner", "convex", "screener", "risk", "exits", "llm"}
_SCANNER_KEYS = {"enabled", "top", "min_price", "min_trades", "min_move_pct"}
_CONVEX_KEYS = {"symbols", "max_expiry_days", "strike_band_pct", "strikes_each_side", "cash_fraction",
                "max_contracts", "max_spread_bps", "max_quote_age_seconds", "min_open_interest",
                "stop_fraction", "take_profit_mult", "entry_start", "entry_end", "time_exit",
                "market_exit", "session_end", "cooldown_seconds", "max_entries_per_day",
                "loop_interval_seconds"}


def _hhmm(section: dict, path: str, key: str):
    """A wall-clock time written as "HH:MM" (Eastern), as a datetime.time."""
    from datetime import time

    value = section[key]
    try:
        hour, minute = str(value).split(":")
        parsed = time(int(hour), int(minute))
    except (TypeError, ValueError):
        parsed = None
    if not isinstance(value, str) or parsed is None:
        _fail(f"{path}.{key}", 'must be a "HH:MM" time (24h, Eastern)', value)
    return parsed
_SIGNAL_KEYS = {"rsi_period", "atr_period", "macd_fast", "macd_slow", "macd_signal",
                "atr_event_mult", "stale_bar_factor", "min_bars",
                "macd_min_hist_atr", "rsi_overbought", "rsi_oversold",
                "trend_ema_fast", "trend_ema_slow",
                "flow_lookback_minutes", "flow_min_trades", "flow_min_imbalance", "flow_exit_bars",
                "tape_event_min_imbalance", "tape_event_min_trades"}
_SCREENER_KEYS = {"min_dte", "max_expiry_lookahead_days", "expiries_to_screen",
                  "strike_band_pct", "otm_only", "min_width_pct", "max_width_pct",
                  "min_open_interest", "max_quote_age_seconds", "max_leg_spread_bps",
                  "min_net_debit", "min_liquid_legs_per_expiry",
                  "min_debit_frac", "max_debit_frac"}
_RISK_KEYS = {"per_entry_fraction", "per_underlying_fraction", "per_cycle_fraction",
              "total_fraction", "allow_stacking", "max_same_direction"}
_EXIT_KEYS = {"stop_fraction", "take_profit_mult", "exit_dte", "reversal_exit", "reversal_needs_flow",
              "trail_arm_mult", "trail_giveback", "flatten_minutes_before_close"}
_LLM_KEYS = {"provider", "base_url", "primary_model", "fallback_models",
             "reasoning_effort", "timeout_seconds", "json_mode"}
_REASONING_EFFORTS = ("low", "high", "max")


def validate(raw: object) -> dict[str, object]:
    """Full check of a parsed settings.yaml; returns {CONSTANT_NAME: value}."""
    if not isinstance(raw, dict):
        raise SettingsError("settings.yaml: the file must be a YAML mapping")
    unknown = set(raw) - _TOP_KEYS
    if unknown:
        _fail("top level", f"unknown key(s) {sorted(unknown)} — a typo?", sorted(unknown))
    missing = _TOP_KEYS - set(raw)
    if missing:
        _fail("top level", f"missing required key(s) {sorted(missing)}", None)

    values: dict[str, object] = {}
    values["SYMBOLS"] = _string_list(raw["symbols"], "symbols")
    if not values["SYMBOLS"]:
        _fail("symbols", "must list at least one symbol", raw["symbols"])
    _, _, bar_seconds = parse_timeframe(raw["bar_timeframe"])
    values["BAR_TIMEFRAME"] = str(raw["bar_timeframe"]).strip()
    values["BAR_SECONDS"] = bar_seconds
    values["LOOP_INTERVAL_SECONDS"] = _integer(raw, "top level", "loop_interval_seconds", 1)

    sig = _section(raw, "signals", _SIGNAL_KEYS)
    values["RSI_PERIOD"] = _integer(sig, "signals", "rsi_period", 1)
    values["ATR_PERIOD"] = _integer(sig, "signals", "atr_period", 1)
    values["MACD_FAST"] = _integer(sig, "signals", "macd_fast", 1)
    values["MACD_SLOW"] = _integer(sig, "signals", "macd_slow", 1)
    values["MACD_SIGNAL"] = _integer(sig, "signals", "macd_signal", 1)
    if values["MACD_FAST"] >= values["MACD_SLOW"]:
        _fail("signals.macd_fast", "must be smaller than macd_slow", sig["macd_fast"])
    values["ATR_EVENT_MULT"] = _number(sig, "signals", "atr_event_mult", 0, lo_open=True)
    values["STALE_BAR_FACTOR"] = _number(sig, "signals", "stale_bar_factor", 1)
    values["MIN_BARS"] = _integer(sig, "signals", "min_bars",
                                  values["MACD_SLOW"] + values["MACD_SIGNAL"])
    values["MACD_MIN_HIST_ATR"] = _number(sig, "signals", "macd_min_hist_atr", 0, 1)
    values["RSI_OVERBOUGHT"] = _number(sig, "signals", "rsi_overbought", 50, 100)
    values["RSI_OVERSOLD"] = _number(sig, "signals", "rsi_oversold", 0, 50)
    if values["RSI_OVERSOLD"] >= values["RSI_OVERBOUGHT"]:
        _fail("signals.rsi_oversold", "must be smaller than rsi_overbought", sig["rsi_oversold"])
    values["TREND_EMA_FAST"] = _integer(sig, "signals", "trend_ema_fast", 1)
    values["TREND_EMA_SLOW"] = _integer(sig, "signals", "trend_ema_slow", 1)
    if values["TREND_EMA_FAST"] >= values["TREND_EMA_SLOW"]:
        _fail("signals.trend_ema_fast", "must be smaller than trend_ema_slow", sig["trend_ema_fast"])
    values["FLOW_LOOKBACK_MINUTES"] = _integer(sig, "signals", "flow_lookback_minutes", 1)
    values["FLOW_MIN_TRADES"] = _integer(sig, "signals", "flow_min_trades", 0)
    values["FLOW_MIN_IMBALANCE"] = _number(sig, "signals", "flow_min_imbalance", 0, 1)
    values["FLOW_EXIT_BARS"] = _integer(sig, "signals", "flow_exit_bars", 1)
    values["TAPE_EVENT_MIN_IMBALANCE"] = _number(sig, "signals", "tape_event_min_imbalance", 0, 1)
    values["TAPE_EVENT_MIN_TRADES"] = _integer(sig, "signals", "tape_event_min_trades", 0)

    scr = _section(raw, "screener", _SCREENER_KEYS)
    values["MIN_DTE"] = _integer(scr, "screener", "min_dte", 1)
    values["MAX_EXPIRY_LOOKAHEAD_DAYS"] = _integer(
        scr, "screener", "max_expiry_lookahead_days", values["MIN_DTE"] + 1)
    values["EXPIRIES_TO_SCREEN"] = _integer(scr, "screener", "expiries_to_screen", 1)
    values["STRIKE_BAND_PCT"] = _number(scr, "screener", "strike_band_pct", 0, 0.5, lo_open=True)
    if not isinstance(scr["otm_only"], bool):
        _fail("screener.otm_only", "must be true or false", scr["otm_only"])
    values["OTM_ONLY"] = scr["otm_only"]
    values["MIN_WIDTH_PCT"] = _number(scr, "screener", "min_width_pct", 0, 0.5, lo_open=True)
    values["MAX_WIDTH_PCT"] = _number(scr, "screener", "max_width_pct", 0, 0.5, lo_open=True)
    if values["MIN_WIDTH_PCT"] > values["MAX_WIDTH_PCT"]:
        _fail("screener.min_width_pct", "must not exceed max_width_pct", scr["min_width_pct"])
    values["MIN_OPEN_INTEREST"] = _integer(scr, "screener", "min_open_interest", 0)
    values["MIN_LIQUID_LEGS_PER_EXPIRY"] = _integer(scr, "screener", "min_liquid_legs_per_expiry", 0)
    values["MAX_QUOTE_AGE_SECONDS"] = _number(scr, "screener", "max_quote_age_seconds", 0, lo_open=True)
    values["MAX_LEG_SPREAD_BPS"] = _number(scr, "screener", "max_leg_spread_bps", 0, lo_open=True)
    values["MIN_NET_DEBIT"] = _number(scr, "screener", "min_net_debit", 0, lo_open=True)
    values["MIN_DEBIT_FRAC"] = _number(scr, "screener", "min_debit_frac", 0, 1, lo_open=True, hi_open=True)
    values["MAX_DEBIT_FRAC"] = _number(scr, "screener", "max_debit_frac", 0, 1, lo_open=True, hi_open=True)
    if values["MIN_DEBIT_FRAC"] > values["MAX_DEBIT_FRAC"]:
        _fail("screener.min_debit_frac", "must not exceed max_debit_frac", scr["min_debit_frac"])

    risk = _section(raw, "risk", _RISK_KEYS)
    values["PER_ENTRY_FRACTION"] = _number(risk, "risk", "per_entry_fraction", 0, 1, lo_open=True)
    values["PER_UNDERLYING_FRACTION"] = _number(risk, "risk", "per_underlying_fraction", 0, 1, lo_open=True)
    values["PER_CYCLE_FRACTION"] = _number(risk, "risk", "per_cycle_fraction", 0, 1, lo_open=True)
    values["TOTAL_FRACTION"] = _number(risk, "risk", "total_fraction", 0, 1, lo_open=True)
    if values["PER_ENTRY_FRACTION"] > values["PER_UNDERLYING_FRACTION"]:
        _fail("risk.per_entry_fraction", "must not exceed per_underlying_fraction",
              risk["per_entry_fraction"])
    if values["PER_UNDERLYING_FRACTION"] > values["TOTAL_FRACTION"]:
        _fail("risk.per_underlying_fraction", "must not exceed total_fraction",
              risk["per_underlying_fraction"])
    if not isinstance(risk["allow_stacking"], bool):
        _fail("risk.allow_stacking", "must be true or false", risk["allow_stacking"])
    values["ALLOW_STACKING"] = risk["allow_stacking"]
    values["MAX_SAME_DIRECTION"] = _integer(risk, "risk", "max_same_direction", 0)

    scanner = _section(raw, "scanner", _SCANNER_KEYS)
    if not isinstance(scanner["enabled"], bool):
        _fail("scanner.enabled", "must be true or false", scanner["enabled"])
    values["SCANNER_ENABLED"] = scanner["enabled"]
    values["SCANNER_TOP"] = _integer(scanner, "scanner", "top", 0)
    values["SCANNER_MIN_PRICE"] = _number(scanner, "scanner", "min_price", 0)
    values["SCANNER_MIN_TRADES"] = _integer(scanner, "scanner", "min_trades", 0)
    values["SCANNER_MIN_MOVE_PCT"] = _number(scanner, "scanner", "min_move_pct", 0)

    convex = _section(raw, "convex", _CONVEX_KEYS)
    values["CONVEX_SYMBOLS"] = _string_list(convex["symbols"], "convex.symbols")
    if not values["CONVEX_SYMBOLS"]:
        _fail("convex.symbols", "must list at least one symbol", convex["symbols"])
    values["CONVEX_MAX_EXPIRY_DAYS"] = _integer(convex, "convex", "max_expiry_days", 0)
    values["CONVEX_STRIKE_BAND_PCT"] = _number(convex, "convex", "strike_band_pct", 0, 0.2, lo_open=True)
    values["CONVEX_STRIKES_EACH_SIDE"] = _integer(convex, "convex", "strikes_each_side", 1)
    values["CONVEX_CASH_FRACTION"] = _number(convex, "convex", "cash_fraction", 0, 1, lo_open=True)
    values["CONVEX_MAX_CONTRACTS"] = _integer(convex, "convex", "max_contracts", 1)
    values["CONVEX_MAX_SPREAD_BPS"] = _number(convex, "convex", "max_spread_bps", 0, lo_open=True)
    values["CONVEX_MAX_QUOTE_AGE_SECONDS"] = _number(convex, "convex", "max_quote_age_seconds", 0, lo_open=True)
    values["CONVEX_MIN_OPEN_INTEREST"] = _integer(convex, "convex", "min_open_interest", 0)
    values["CONVEX_STOP_FRACTION"] = _number(convex, "convex", "stop_fraction", 0, 1, lo_open=True, hi_open=True)
    values["CONVEX_TAKE_PROFIT_MULT"] = _number(convex, "convex", "take_profit_mult", 1, lo_open=True)
    clock_keys = ("entry_start", "entry_end", "time_exit", "market_exit", "session_end")
    times = [_hhmm(convex, "convex", key) for key in clock_keys]
    for earlier, later, key in zip(times, times[1:], clock_keys[1:]):
        if later <= earlier:
            _fail(f"convex.{key}", "must be later than the previous convex time", convex[key])
    for key, value in zip(clock_keys, times):
        values[f"CONVEX_{key.upper()}"] = value
    values["CONVEX_COOLDOWN_SECONDS"] = _integer(convex, "convex", "cooldown_seconds", 0)
    values["CONVEX_MAX_ENTRIES_PER_DAY"] = _integer(convex, "convex", "max_entries_per_day", 1)
    values["CONVEX_LOOP_INTERVAL_SECONDS"] = _integer(convex, "convex", "loop_interval_seconds", 5)

    exits = _section(raw, "exits", _EXIT_KEYS)
    values["STOP_FRACTION"] = _number(exits, "exits", "stop_fraction", 0, 1, lo_open=True, hi_open=True)
    values["TAKE_PROFIT_MULT"] = _number(exits, "exits", "take_profit_mult", 1, lo_open=True)
    values["EXIT_DTE"] = _integer(exits, "exits", "exit_dte", 0)
    values["FLATTEN_MINUTES_BEFORE_CLOSE"] = _integer(exits, "exits", "flatten_minutes_before_close", 0)
    if not isinstance(exits["reversal_exit"], bool):
        _fail("exits.reversal_exit", "must be true or false", exits["reversal_exit"])
    values["REVERSAL_EXIT"] = exits["reversal_exit"]
    if not isinstance(exits["reversal_needs_flow"], bool):
        _fail("exits.reversal_needs_flow", "must be true or false", exits["reversal_needs_flow"])
    values["REVERSAL_NEEDS_FLOW"] = exits["reversal_needs_flow"]
    arm = _number(exits, "exits", "trail_arm_mult", 0)
    if 0 < arm < 1:
        _fail("exits.trail_arm_mult", "must be 0 (off) or at least 1 (a multiple of the entry debit)", exits["trail_arm_mult"])
    values["TRAIL_ARM_MULT"] = arm
    values["TRAIL_GIVEBACK"] = _number(exits, "exits", "trail_giveback", 0, 1, lo_open=True, hi_open=True)

    llm = _section(raw, "llm", _LLM_KEYS)
    provider = llm["provider"]
    if not isinstance(provider, str) or not provider.strip():
        _fail("llm.provider", "must be a non-empty provider name", provider)
    values["LLM_PROVIDER"] = provider.strip()
    base_url = llm["base_url"]
    if not isinstance(base_url, str) or not base_url.strip().startswith("http"):
        _fail("llm.base_url", "must be an http(s) URL such as https://api.featherless.ai/v1", base_url)
    values["LLM_BASE_URL"] = base_url.strip().rstrip("/")
    primary = llm["primary_model"]
    if not isinstance(primary, str) or not primary.strip():
        _fail("llm.primary_model", "must be a non-empty model name", primary)
    values["PRIMARY_MODEL"] = primary.strip()
    fallbacks = llm["fallback_models"]
    if not isinstance(fallbacks, list) or any(not isinstance(m, str) or not m.strip() for m in fallbacks):
        _fail("llm.fallback_models", "must be a list of model names (may be empty)", fallbacks)
    values["FALLBACK_MODELS"] = tuple(m.strip() for m in fallbacks)
    effort = llm["reasoning_effort"]
    if not isinstance(effort, str) or effort.strip().lower() not in _REASONING_EFFORTS:
        _fail("llm.reasoning_effort", f"must be one of {list(_REASONING_EFFORTS)}", effort)
    values["LLM_REASONING_EFFORT"] = effort.strip().lower()
    values["LLM_TIMEOUT_SECONDS"] = _number(llm, "llm", "timeout_seconds", 5, 120)
    if not isinstance(llm["json_mode"], bool):
        _fail("llm.json_mode", "must be true or false", llm["json_mode"])
    values["LLM_JSON_MODE"] = llm["json_mode"]

    return values


def load_settings(path: Path = SETTINGS_PATH) -> dict[str, object]:
    """Parse + validate the YAML file; raises SettingsError with the exact problem."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise SettingsError(f"cannot read {path}: {type(error).__name__}") from None
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise SettingsError(f"{path} is not valid YAML: {error}") from None
    return validate(raw)


# Loaded once at import: a bad settings.yaml stops every command immediately.
_VALUES = load_settings()
SYMBOLS: tuple[str, ...] = _VALUES["SYMBOLS"]  # type: ignore[assignment]
BAR_TIMEFRAME: str = _VALUES["BAR_TIMEFRAME"]  # type: ignore[assignment]
BAR_SECONDS: int = _VALUES["BAR_SECONDS"]  # type: ignore[assignment]
LOOP_INTERVAL_SECONDS: int = _VALUES["LOOP_INTERVAL_SECONDS"]  # type: ignore[assignment]
RSI_PERIOD: int = _VALUES["RSI_PERIOD"]  # type: ignore[assignment]
ATR_PERIOD: int = _VALUES["ATR_PERIOD"]  # type: ignore[assignment]
MACD_FAST: int = _VALUES["MACD_FAST"]  # type: ignore[assignment]
MACD_SLOW: int = _VALUES["MACD_SLOW"]  # type: ignore[assignment]
MACD_SIGNAL: int = _VALUES["MACD_SIGNAL"]  # type: ignore[assignment]
ATR_EVENT_MULT: float = _VALUES["ATR_EVENT_MULT"]  # type: ignore[assignment]
STALE_BAR_FACTOR: float = _VALUES["STALE_BAR_FACTOR"]  # type: ignore[assignment]
MIN_BARS: int = _VALUES["MIN_BARS"]  # type: ignore[assignment]
MACD_MIN_HIST_ATR: float = _VALUES["MACD_MIN_HIST_ATR"]  # type: ignore[assignment]
RSI_OVERBOUGHT: float = _VALUES["RSI_OVERBOUGHT"]  # type: ignore[assignment]
RSI_OVERSOLD: float = _VALUES["RSI_OVERSOLD"]  # type: ignore[assignment]
TREND_EMA_FAST: int = _VALUES["TREND_EMA_FAST"]  # type: ignore[assignment]
TREND_EMA_SLOW: int = _VALUES["TREND_EMA_SLOW"]  # type: ignore[assignment]
FLOW_LOOKBACK_MINUTES: int = _VALUES["FLOW_LOOKBACK_MINUTES"]  # type: ignore[assignment]
FLOW_MIN_TRADES: int = _VALUES["FLOW_MIN_TRADES"]  # type: ignore[assignment]
FLOW_MIN_IMBALANCE: float = _VALUES["FLOW_MIN_IMBALANCE"]  # type: ignore[assignment]
FLOW_EXIT_BARS: int = _VALUES["FLOW_EXIT_BARS"]  # type: ignore[assignment]
TAPE_EVENT_MIN_IMBALANCE: float = _VALUES["TAPE_EVENT_MIN_IMBALANCE"]  # type: ignore[assignment]
TAPE_EVENT_MIN_TRADES: int = _VALUES["TAPE_EVENT_MIN_TRADES"]  # type: ignore[assignment]
MIN_DTE: int = _VALUES["MIN_DTE"]  # type: ignore[assignment]
MAX_EXPIRY_LOOKAHEAD_DAYS: int = _VALUES["MAX_EXPIRY_LOOKAHEAD_DAYS"]  # type: ignore[assignment]
EXPIRIES_TO_SCREEN: int = _VALUES["EXPIRIES_TO_SCREEN"]  # type: ignore[assignment]
STRIKE_BAND_PCT: float = _VALUES["STRIKE_BAND_PCT"]  # type: ignore[assignment]
OTM_ONLY: bool = _VALUES["OTM_ONLY"]  # type: ignore[assignment]
MIN_WIDTH_PCT: float = _VALUES["MIN_WIDTH_PCT"]  # type: ignore[assignment]
MAX_WIDTH_PCT: float = _VALUES["MAX_WIDTH_PCT"]  # type: ignore[assignment]
MIN_OPEN_INTEREST: int = _VALUES["MIN_OPEN_INTEREST"]  # type: ignore[assignment]
MIN_LIQUID_LEGS_PER_EXPIRY: int = _VALUES["MIN_LIQUID_LEGS_PER_EXPIRY"]  # type: ignore[assignment]
MAX_QUOTE_AGE_SECONDS: float = _VALUES["MAX_QUOTE_AGE_SECONDS"]  # type: ignore[assignment]
MAX_LEG_SPREAD_BPS: float = _VALUES["MAX_LEG_SPREAD_BPS"]  # type: ignore[assignment]
MIN_NET_DEBIT: float = _VALUES["MIN_NET_DEBIT"]  # type: ignore[assignment]
MIN_DEBIT_FRAC: float = _VALUES["MIN_DEBIT_FRAC"]  # type: ignore[assignment]
MAX_DEBIT_FRAC: float = _VALUES["MAX_DEBIT_FRAC"]  # type: ignore[assignment]
PER_ENTRY_FRACTION: float = _VALUES["PER_ENTRY_FRACTION"]  # type: ignore[assignment]
PER_UNDERLYING_FRACTION: float = _VALUES["PER_UNDERLYING_FRACTION"]  # type: ignore[assignment]
PER_CYCLE_FRACTION: float = _VALUES["PER_CYCLE_FRACTION"]  # type: ignore[assignment]
TOTAL_FRACTION: float = _VALUES["TOTAL_FRACTION"]  # type: ignore[assignment]
ALLOW_STACKING: bool = _VALUES["ALLOW_STACKING"]  # type: ignore[assignment]
MAX_SAME_DIRECTION: int = _VALUES["MAX_SAME_DIRECTION"]  # type: ignore[assignment]
STOP_FRACTION: float = _VALUES["STOP_FRACTION"]  # type: ignore[assignment]
TAKE_PROFIT_MULT: float = _VALUES["TAKE_PROFIT_MULT"]  # type: ignore[assignment]
EXIT_DTE: int = _VALUES["EXIT_DTE"]  # type: ignore[assignment]
FLATTEN_MINUTES_BEFORE_CLOSE: int = _VALUES["FLATTEN_MINUTES_BEFORE_CLOSE"]  # type: ignore[assignment]
SCANNER_ENABLED: bool = _VALUES["SCANNER_ENABLED"]  # type: ignore[assignment]
SCANNER_TOP: int = _VALUES["SCANNER_TOP"]  # type: ignore[assignment]
SCANNER_MIN_PRICE: float = _VALUES["SCANNER_MIN_PRICE"]  # type: ignore[assignment]
SCANNER_MIN_TRADES: int = _VALUES["SCANNER_MIN_TRADES"]  # type: ignore[assignment]
SCANNER_MIN_MOVE_PCT: float = _VALUES["SCANNER_MIN_MOVE_PCT"]  # type: ignore[assignment]
CONVEX_SYMBOLS: tuple[str, ...] = _VALUES["CONVEX_SYMBOLS"]  # type: ignore[assignment]
CONVEX_MAX_EXPIRY_DAYS: int = _VALUES["CONVEX_MAX_EXPIRY_DAYS"]  # type: ignore[assignment]
CONVEX_STRIKE_BAND_PCT: float = _VALUES["CONVEX_STRIKE_BAND_PCT"]  # type: ignore[assignment]
CONVEX_STRIKES_EACH_SIDE: int = _VALUES["CONVEX_STRIKES_EACH_SIDE"]  # type: ignore[assignment]
CONVEX_CASH_FRACTION: float = _VALUES["CONVEX_CASH_FRACTION"]  # type: ignore[assignment]
CONVEX_MAX_CONTRACTS: int = _VALUES["CONVEX_MAX_CONTRACTS"]  # type: ignore[assignment]
CONVEX_MAX_SPREAD_BPS: float = _VALUES["CONVEX_MAX_SPREAD_BPS"]  # type: ignore[assignment]
CONVEX_MAX_QUOTE_AGE_SECONDS: float = _VALUES["CONVEX_MAX_QUOTE_AGE_SECONDS"]  # type: ignore[assignment]
CONVEX_MIN_OPEN_INTEREST: int = _VALUES["CONVEX_MIN_OPEN_INTEREST"]  # type: ignore[assignment]
CONVEX_STOP_FRACTION: float = _VALUES["CONVEX_STOP_FRACTION"]  # type: ignore[assignment]
CONVEX_TAKE_PROFIT_MULT: float = _VALUES["CONVEX_TAKE_PROFIT_MULT"]  # type: ignore[assignment]
CONVEX_ENTRY_START = _VALUES["CONVEX_ENTRY_START"]
CONVEX_ENTRY_END = _VALUES["CONVEX_ENTRY_END"]
CONVEX_TIME_EXIT = _VALUES["CONVEX_TIME_EXIT"]
CONVEX_MARKET_EXIT = _VALUES["CONVEX_MARKET_EXIT"]
CONVEX_SESSION_END = _VALUES["CONVEX_SESSION_END"]
CONVEX_COOLDOWN_SECONDS: int = _VALUES["CONVEX_COOLDOWN_SECONDS"]  # type: ignore[assignment]
CONVEX_MAX_ENTRIES_PER_DAY: int = _VALUES["CONVEX_MAX_ENTRIES_PER_DAY"]  # type: ignore[assignment]
CONVEX_LOOP_INTERVAL_SECONDS: int = _VALUES["CONVEX_LOOP_INTERVAL_SECONDS"]  # type: ignore[assignment]
REVERSAL_EXIT: bool = _VALUES["REVERSAL_EXIT"]  # type: ignore[assignment]
REVERSAL_NEEDS_FLOW: bool = _VALUES["REVERSAL_NEEDS_FLOW"]  # type: ignore[assignment]
TRAIL_ARM_MULT: float = _VALUES["TRAIL_ARM_MULT"]  # type: ignore[assignment]
TRAIL_GIVEBACK: float = _VALUES["TRAIL_GIVEBACK"]  # type: ignore[assignment]
LLM_PROVIDER: str = _VALUES["LLM_PROVIDER"]  # type: ignore[assignment]
LLM_BASE_URL: str = _VALUES["LLM_BASE_URL"]  # type: ignore[assignment]
PRIMARY_MODEL: str = _VALUES["PRIMARY_MODEL"]  # type: ignore[assignment]
FALLBACK_MODELS: tuple[str, ...] = _VALUES["FALLBACK_MODELS"]  # type: ignore[assignment]
LLM_REASONING_EFFORT: str = _VALUES["LLM_REASONING_EFFORT"]  # type: ignore[assignment]
LLM_TIMEOUT_SECONDS: float = _VALUES["LLM_TIMEOUT_SECONDS"]  # type: ignore[assignment]
LLM_JSON_MODE: bool = _VALUES["LLM_JSON_MODE"]  # type: ignore[assignment]
