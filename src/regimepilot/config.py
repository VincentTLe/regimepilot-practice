"""Configuration loading and paper-trading safety guards.

Credentials are read from environment variables only, are held inside
pydantic ``SecretStr``, and are never printed, logged or embedded in an error
message.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# The paper host is the only Alpaca trading host this project may ever touch.
PAPER_TRADING_HOST = "paper-api.alpaca.markets"
LIVE_TRADING_HOST = "api.alpaca.markets"

TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
FALSE_VALUES = frozenset({"false", "0", "no", "off"})

# Environment variables that would explicitly ask for live trading.
LIVE_FLAG_VARS = ("ALPACA_LIVE", "ALPACA_LIVE_TRADING", "APCA_LIVE")

# Environment variables that could point an Alpaca SDK at a non-paper endpoint.
ENDPOINT_VARS = (
    "ALPACA_BASE_URL",
    "ALPACA_API_BASE_URL",
    "APCA_API_BASE_URL",
    "ALPACA_ENDPOINT",
)


class ConfigError(RuntimeError):
    """Configuration is missing, invalid or unsafe.

    Messages raised here are printed to the user, so they must never contain a
    credential value.
    """


class Settings(BaseSettings):
    """Raw ``ALPACA_*`` values read from the environment and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    alpaca_api_key: SecretStr = SecretStr("")
    alpaca_secret_key: SecretStr = SecretStr("")

    # Kept as a raw string so load_settings() can reject a typo with a clear
    # message instead of letting pydantic quietly coerce it.
    alpaca_paper: str = "true"

    # Declared only so the guards below can see them if they are set anywhere.
    alpaca_live: str = ""
    alpaca_live_trading: str = ""
    apca_live: str = ""
    alpaca_base_url: str = ""
    alpaca_api_base_url: str = ""
    apca_api_base_url: str = ""
    alpaca_endpoint: str = ""

    @property
    def paper(self) -> bool:
        return parse_bool("ALPACA_PAPER", self.alpaca_paper)

    def __repr__(self) -> str:
        # Explicit, so that a stray print() or a traceback can never surface a key.
        return f"Settings(alpaca_api_key=<hidden>, alpaca_secret_key=<hidden>, alpaca_paper={self.alpaca_paper!r})"

    __str__ = __repr__


_FIELD_TO_ENV_VAR = {field: field.upper() for field in Settings.model_fields}

_CREDENTIAL_FIELDS = ("alpaca_api_key", "alpaca_secret_key")


def parse_bool(name: str, value: str) -> bool:
    """Parse a strict boolean environment value.

    Anything unrecognised raises rather than defaulting, so a typo can never be
    read as "safe".
    """
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    allowed = ", ".join(sorted(TRUE_VALUES | FALSE_VALUES))
    raise ConfigError(f"{name} must be one of: {allowed}. Got {value!r}.")


def find_live_trading_signals(env: Mapping[str, str]) -> list[str]:
    """Return every reason ``env`` looks like live trading. An empty list means safe."""
    signals: list[str] = []

    if not parse_bool("ALPACA_PAPER", env.get("ALPACA_PAPER", "true")):
        signals.append("ALPACA_PAPER is false, but this project is paper-only.")

    for name in LIVE_FLAG_VARS:
        if str(env.get(name, "")).strip().lower() in TRUE_VALUES:
            signals.append(f"{name} is set to a true value, but live trading is never allowed.")

    for name in ENDPOINT_VARS:
        url = str(env.get(name, "")).strip().lower()
        # The paper host contains the live host as a substring, so check it first.
        if not url or PAPER_TRADING_HOST in url:
            continue
        if LIVE_TRADING_HOST in url:
            signals.append(f"{name} points at the live trading endpoint {LIVE_TRADING_HOST}.")

    return signals


def _guard_mapping(settings: Settings) -> dict[str, str]:
    """Flatten the non-secret settings back into plain env values for the guards."""
    return {
        var: getattr(settings, field)
        for field, var in _FIELD_TO_ENV_VAR.items()
        if field not in _CREDENTIAL_FIELDS
    }


def settings_from_mapping(env: Mapping[str, str]) -> Settings:
    """Build Settings from an explicit mapping and nothing else.

    Passing ``_env_file=None`` plus a value for every field keeps the real
    process environment and any local .env file out of unit tests.
    """
    values = {
        field: env.get(var, Settings.model_fields[field].default)
        for field, var in _FIELD_TO_ENV_VAR.items()
    }
    return Settings(_env_file=None, **values)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Load configuration and refuse anything that is not paper trading.

    ``env=None`` reads the process environment plus a local .env file. Tests
    pass an explicit mapping so no real credential is ever touched.
    """
    settings = Settings() if env is None else settings_from_mapping(env)

    signals = find_live_trading_signals(_guard_mapping(settings))
    if signals:
        raise ConfigError("Refusing to run. " + " ".join(signals))

    missing = [
        _FIELD_TO_ENV_VAR[field]
        for field in _CREDENTIAL_FIELDS
        if not getattr(settings, field).get_secret_value().strip()
    ]
    if missing:
        raise ConfigError(
            f"Missing credentials: {', '.join(missing)}. "
            "Copy .env.example to .env and fill it in locally."
        )

    return settings
