"""Config tests. Every case passes an explicit env mapping, so no real .env
file, no real credential and no network is ever touched."""

import pytest

from regimepilot.config import (
    ConfigError,
    find_live_trading_signals,
    load_settings,
    parse_bool,
)

PAPER_ENV = {
    "ALPACA_API_KEY": "test-key",
    "ALPACA_SECRET_KEY": "test-secret",
    "ALPACA_PAPER": "true",
}


def test_valid_paper_config_loads():
    settings = load_settings(PAPER_ENV)
    assert settings.paper is True
    assert settings.alpaca_api_key.get_secret_value() == "test-key"


@pytest.mark.parametrize("blanked", ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"])
def test_blank_credential_is_rejected(blanked):
    env = dict(PAPER_ENV, **{blanked: "   "})
    with pytest.raises(ConfigError) as excinfo:
        load_settings(env)
    assert blanked in str(excinfo.value)


def test_absent_credentials_are_rejected():
    with pytest.raises(ConfigError) as excinfo:
        load_settings({"ALPACA_PAPER": "true"})
    assert "ALPACA_API_KEY" in str(excinfo.value)
    assert "ALPACA_SECRET_KEY" in str(excinfo.value)


@pytest.mark.parametrize("value", ["maybe", "", "2", "paper", "True!"])
def test_invalid_alpaca_paper_value_is_rejected(value):
    env = dict(PAPER_ENV, ALPACA_PAPER=value)
    with pytest.raises(ConfigError) as excinfo:
        load_settings(env)
    assert "ALPACA_PAPER" in str(excinfo.value)


def test_alpaca_paper_false_is_rejected():
    env = dict(PAPER_ENV, ALPACA_PAPER="false")
    with pytest.raises(ConfigError) as excinfo:
        load_settings(env)
    assert "paper-only" in str(excinfo.value)


@pytest.mark.parametrize("var", ["ALPACA_LIVE", "ALPACA_LIVE_TRADING", "APCA_LIVE"])
def test_live_trading_flag_is_rejected(var):
    env = dict(PAPER_ENV, **{var: "true"})
    with pytest.raises(ConfigError) as excinfo:
        load_settings(env)
    assert var in str(excinfo.value)


@pytest.mark.parametrize(
    "var",
    ["ALPACA_BASE_URL", "ALPACA_API_BASE_URL", "APCA_API_BASE_URL", "ALPACA_ENDPOINT"],
)
def test_live_endpoint_is_rejected(var):
    env = dict(PAPER_ENV, **{var: "https://api.alpaca.markets/v2"})
    with pytest.raises(ConfigError) as excinfo:
        load_settings(env)
    assert var in str(excinfo.value)


def test_paper_endpoint_is_allowed():
    env = dict(PAPER_ENV, ALPACA_BASE_URL="https://paper-api.alpaca.markets")
    assert load_settings(env).paper is True


def test_market_data_endpoint_is_not_mistaken_for_live():
    env = dict(PAPER_ENV, ALPACA_ENDPOINT="https://data.alpaca.markets/v2")
    assert load_settings(env).paper is True


def test_clean_paper_env_produces_no_live_signals():
    assert find_live_trading_signals(PAPER_ENV) == []


def test_settings_never_reveal_credentials():
    settings = load_settings(PAPER_ENV)
    rendered = f"{settings!r} {settings!s}"
    assert "test-key" not in rendered
    assert "test-secret" not in rendered


def test_config_error_for_missing_credentials_omits_values():
    env = dict(PAPER_ENV, ALPACA_SECRET_KEY="")
    with pytest.raises(ConfigError) as excinfo:
        load_settings(env)
    assert "test-key" not in str(excinfo.value)


def test_parse_bool_accepts_common_spellings():
    assert parse_bool("X", "TRUE") is True
    assert parse_bool("X", " On ") is True
    assert parse_bool("X", " off ") is False
    assert parse_bool("X", "0") is False
