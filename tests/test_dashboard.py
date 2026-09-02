"""dashboard.py: exports must never raise into the trading loop, and never leak the llm section."""

import json

import yaml

import dashboard


def test_config_json_strips_llm_section(tmp_path):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        yaml.safe_dump({"symbols": ["SPY"], "llm": {"base_url": "https://x", "primary_model": "m"}}),
        encoding="utf-8",
    )
    out = tmp_path / "config.json"
    dashboard.write_config_json(settings_path, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == {"symbols": ["SPY"]}


def test_cli_snapshot_falls_back_when_binary_missing(monkeypatch):
    monkeypatch.setattr(dashboard.shutil, "which", lambda name: None)
    snapshot = dashboard.cli_snapshot(profile=None)
    assert snapshot["source"] == "sdk"
    assert snapshot["cli_error"] == "cli_not_found"


def test_cli_snapshot_parses_cli_json(monkeypatch):
    monkeypatch.setattr(dashboard.shutil, "which", lambda name: "alpaca")
    answers = {
        "clock": json.dumps({"is_open": True, "timestamp": "2026-09-03T14:00:00Z"}),
        "account": json.dumps({"account_number": "PA123", "equity": "100500.5", "status": "ACTIVE",
                               "buying_power": "200000", "secret_looking_field": "x"}),
        "position": json.dumps([{"symbol": "NVDA260911C00230000", "qty": "11"}]),
    }

    def fake_run(args, timeout=None):
        return answers[args[1]]

    monkeypatch.setattr(dashboard, "_run", fake_run)
    snapshot = dashboard.cli_snapshot(profile="tape")
    assert snapshot["source"] == "alpaca-cli"
    assert snapshot["clock"]["is_open"] is True
    assert snapshot["account"] == {"account_number": "PA123", "equity": "100500.5",
                                   "status": "ACTIVE", "buying_power": "200000"}
    assert snapshot["positions"] == [{"symbol": "NVDA260911C00230000", "qty": "11"}]


def test_export_all_reports_failures_without_raising(monkeypatch, tmp_path):
    def boom(args, timeout=None):
        raise dashboard.DashboardError("subprocess failed: CalledProcessError")

    monkeypatch.setattr(dashboard, "_run", boom)
    monkeypatch.setattr(dashboard.shutil, "which", lambda name: None)
    statuses = dashboard.export_all(tmp_path, candles=True, deploy_enabled=False)
    assert set(statuses) >= {"account", "pnl", "config", "journal", "cli_snapshot", "candles"}
    assert all(isinstance(v, str) for v in statuses.values())
