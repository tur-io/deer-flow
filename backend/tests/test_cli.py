from pathlib import Path

import yaml

from src import cli


def _write_config(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_models_auth_login_creates_openai_codex_model(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, {"models": []})

    exit_code = cli.main(["models", "auth", "login", "--provider", "openai-codex", "--config", str(config_path)])

    assert exit_code == 0
    data = _read_config(config_path)
    assert len(data["models"]) == 1
    model = data["models"][0]
    assert model["name"] == "openai-codex"
    assert model["use"] == "langchain_openai:ChatOpenAI"
    assert model["api_key"] == "$OPENAI_API_KEY"


def test_models_auth_login_updates_and_sets_default(tmp_path):
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
