from pathlib import Path

import yaml

from src import cli


def _write_config(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_models_auth_login_creates_openai_codex_model(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, {"models": []})
    monkeypatch.setattr(cli, "_login_openai_codex_oauth", lambda **_: {"refresh_token": "rt", "client_id": "cid"})

    exit_code = cli.main(["models", "auth", "login", "--provider", "openai-codex", "--config", str(config_path)])

    assert exit_code == 0
    data = _read_config(config_path)
    assert len(data["models"]) == 1
    model = data["models"][0]
    assert model["name"] == "openai-codex"
    assert model["use"] == "langchain_openai:ChatOpenAI"
    assert model["oauth"]["refresh_token"] == "rt"


def test_models_auth_login_updates_and_sets_default(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        {
            "models": [
                {"name": "gpt-4", "use": "langchain_openai:ChatOpenAI", "model": "gpt-4", "api_key": "$OPENAI_API_KEY"},
                {"name": "openai-codex", "use": "legacy", "model": "legacy", "api_key": "legacy"},
            ]
        },
    )
    monkeypatch.setattr(cli, "_login_openai_codex_oauth", lambda **_: {"refresh_token": "rt", "client_id": "cid"})

    exit_code = cli.main(["models", "auth", "login", "--provider", "openai-codex", "--set-default", "--config", str(config_path)])

    assert exit_code == 0
    data = _read_config(config_path)
    assert data["models"][0]["name"] == "openai-codex"
    assert data["models"][0]["model"] == "gpt-5"
    assert data["models"][1]["name"] == "gpt-4"


def test_models_auth_login_rejects_unknown_provider(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, {"models": []})

    try:
        cli.main(["models", "auth", "login", "--provider", "unknown", "--config", str(config_path)])
    except ValueError as exc:
        assert "Unsupported provider" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported provider")


def test_models_auth_login_accepts_models_object_mapping(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        {
            "models": {
                "gpt-4": {"use": "langchain_openai:ChatOpenAI", "model": "gpt-4", "api_key": "$OPENAI_API_KEY"}
            }
        },
    )
    monkeypatch.setattr(cli, "_login_openai_codex_oauth", lambda **_: {"refresh_token": "rt", "client_id": "cid"})

    exit_code = cli.main(["models", "auth", "login", "--provider", "openai-codex", "--set-default", "--config", str(config_path)])

    assert exit_code == 0
    data = _read_config(config_path)
    assert isinstance(data["models"], list)
    assert data["models"][0]["name"] == "openai-codex"
    assert any(model["name"] == "gpt-4" for model in data["models"])


def test_models_auth_login_accepts_empty_models_object(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, {"models": {}})
    monkeypatch.setattr(cli, "_login_openai_codex_oauth", lambda **_: {"refresh_token": "rt", "client_id": "cid"})

    exit_code = cli.main(["models", "auth", "login", "--provider", "openai-codex", "--config", str(config_path)])

    assert exit_code == 0
    data = _read_config(config_path)
    assert len(data["models"]) == 1
    assert data["models"][0]["name"] == "openai-codex"


def test_models_auth_login_prints_token_setup_steps_when_env_missing(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, {"models": []})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = cli.main(["models", "auth", "login", "--provider", "openai-codex", "--no-oauth", "--config", str(config_path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "https://platform.openai.com/api-keys" in out
    assert "export OPENAI_API_KEY='sk-...'" in out


def test_models_auth_login_detects_existing_openai_key(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, {"models": []})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-1234567890")

    exit_code = cli.main(["models", "auth", "login", "--provider", "openai-codex", "--no-oauth", "--config", str(config_path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Detected OPENAI_API_KEY in environment" in out
    assert "ready to start DeerFlow" in out


def test_models_auth_login_runs_oauth_and_writes_refresh_token(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, {"models": []})
    monkeypatch.setattr(cli, "_login_openai_codex_oauth", lambda **_: {"refresh_token": "refresh-token", "client_id": "app_test"})

    exit_code = cli.main(["models", "auth", "login", "--provider", "openai-codex", "--config", str(config_path)])

    assert exit_code == 0
    data = _read_config(config_path)
    model = data["models"][0]
    assert model["oauth"]["token_url"] == "https://auth.openai.com/oauth/token"
    assert model["oauth"]["grant_type"] == "refresh_token"
    assert model["oauth"]["refresh_token"] == "refresh-token"
    assert model["oauth"]["client_id"] == "app_test"

    out = capsys.readouterr().out
    assert "OAuth token saved" in out


def test_parse_redirect_url_for_code_rejects_state_mismatch():
    redirect = "http://localhost:1455/auth/callback?code=abc&state=bad"
    try:
        cli._parse_redirect_url_for_code(redirect_url=redirect, expected_state="good")
    except ValueError as exc:
        assert "state mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
