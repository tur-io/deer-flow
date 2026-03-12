from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from src.config.app_config import AppConfig

OPENAI_CODEX_PROVIDER = "openai-codex"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML object: {path}")
    return data


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _ensure_models_list(config_data: dict[str, Any]) -> list[dict[str, Any]]:
    models = config_data.get("models")

    if models is None:
        models = []
    elif isinstance(models, dict):
        nested_models = models.get("models") if isinstance(models.get("models"), list) else None
        nested_providers = models.get("providers") if isinstance(models.get("providers"), list) else None

        if nested_models is not None:
            models = nested_models
        elif nested_providers is not None:
            models = nested_providers
        else:
            converted_models: list[dict[str, Any]] = []
            for name, item in models.items():
                if not isinstance(item, dict):
                    continue
                converted = dict(item)
                converted.setdefault("name", name)
                converted_models.append(converted)
            models = converted_models

    if not isinstance(models, list):
        raise ValueError("'models' must be a list or object in config.yaml")

    for idx, item in enumerate(models):
        if not isinstance(item, dict):
            raise ValueError(f"'models[{idx}]' must be a YAML object")

    config_data["models"] = models
    return models


def _upsert_openai_codex_model(config_data: dict[str, Any], *, set_default: bool) -> tuple[bool, str]:
    models = _ensure_models_list(config_data)

    model_entry = {
        "name": OPENAI_CODEX_PROVIDER,
        "display_name": "OpenAI Codex",
        "use": "langchain_openai:ChatOpenAI",
        "model": "gpt-5",
        "api_key": "$OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "supports_vision": True,
        "supports_reasoning_effort": True,
    }

    existing_index = next((i for i, item in enumerate(models) if isinstance(item, dict) and item.get("name") == OPENAI_CODEX_PROVIDER), None)
    created = existing_index is None

    if created:
        models.append(model_entry)
        existing_index = len(models) - 1
    else:
        assert existing_index is not None
        existing = models[existing_index]
        if not isinstance(existing, dict):
            raise ValueError(f"Model '{OPENAI_CODEX_PROVIDER}' must be a YAML object")
        existing.update(model_entry)

    if set_default and existing_index != 0:
        selected = models.pop(existing_index)
        models.insert(0, selected)

    action = "Created" if created else "Updated"
    return created, f"{action} model '{OPENAI_CODEX_PROVIDER}' in config.yaml"


def _print_openai_codex_auth_next_steps() -> None:
    existing_key = os.getenv("OPENAI_API_KEY")
    if existing_key:
        masked = f"{existing_key[:7]}..." if len(existing_key) > 10 else "(set)"
        print(f"Detected OPENAI_API_KEY in environment: {masked}")
        print("You're ready to start DeerFlow with OpenAI Codex.")
        return

    print("Next step: create an OpenAI API key and set OPENAI_API_KEY before starting DeerFlow.")
    print("1) Create a key at: https://platform.openai.com/api-keys")
    print("2) Set it for your current shell:")
    print("   export OPENAI_API_KEY='sk-...'")
    print("3) Optional: persist it in a local .env file:")
    print("   echo \"OPENAI_API_KEY=sk-...\" >> .env")


def _cmd_models_auth_login(args: argparse.Namespace) -> int:
    if args.provider != OPENAI_CODEX_PROVIDER:
        raise ValueError(f"Unsupported provider '{args.provider}'. Currently supported: {OPENAI_CODEX_PROVIDER}")

    config_path = AppConfig.resolve_config_path(args.config)
    config_data = _load_yaml(config_path)
    _, message = _upsert_openai_codex_model(config_data, set_default=args.set_default)
    _save_yaml(config_path, config_data)

    print(message)
    print(f"Config updated: {config_path}")
    _print_openai_codex_auth_next_steps()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deerflow", description="DeerFlow CLI utilities")
    subparsers = parser.add_subparsers(dest="command")

    models_parser = subparsers.add_parser("models", help="Model management commands")
    models_subparsers = models_parser.add_subparsers(dest="models_command")

    auth_parser = models_subparsers.add_parser("auth", help="Model auth commands")
    auth_subparsers = auth_parser.add_subparsers(dest="auth_command")

    login_parser = auth_subparsers.add_parser("login", help="Configure model auth provider")
    login_parser.add_argument("--provider", required=True, help="Provider to configure")
    login_parser.add_argument("--set-default", action="store_true", help="Move this provider to the top of models list")
    login_parser.add_argument("--config", default=None, help="Path to config.yaml (optional)")
    login_parser.set_defaults(func=_cmd_models_auth_login)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1

    return func(args)


if __name__ == "__main__":
    raise SystemExit(main())
