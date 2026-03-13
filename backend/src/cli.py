from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import threading
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import yaml

from src.config.app_config import AppConfig

OPENAI_CODEX_PROVIDER = "openai-codex"
OPENAI_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_REDIRECT_URI = "http://localhost:1455/auth/callback"
OPENAI_CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_CODEX_SCOPE = "openid profile email offline_access"


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    server: _OAuthHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_response(404)
            self.end_headers()
            return

        query = urllib.parse.parse_qs(parsed.query)
        self.server.callback_query = {key: values[0] for key, values in query.items() if values}

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"<html><body><h2>OpenAI OAuth complete. You can close this tab.</h2></body></html>")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class _OAuthHTTPServer(HTTPServer):
    callback_query: dict[str, str] | None


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


def _upsert_openai_codex_model(config_data: dict[str, Any], *, set_default: bool, oauth_credential: dict[str, Any] | None = None) -> tuple[bool, str]:
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

    if oauth_credential is not None:
        oauth_block: dict[str, Any] = {
            "enabled": True,
            "token_url": OPENAI_CODEX_OAUTH_TOKEN_URL,
            "grant_type": "refresh_token",
            "refresh_token": oauth_credential["refresh_token"],
        }
        client_id = oauth_credential.get("client_id")
        if isinstance(client_id, str) and client_id.strip():
            oauth_block["client_id"] = client_id
        model_entry["oauth"] = oauth_block

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
    print('   echo "OPENAI_API_KEY=sk-..." >> .env')


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _build_openai_codex_authorize_url(*, state: str, code_challenge: str, redirect_uri: str) -> str:
    params = {
        "response_type": "code",
        "client_id": OPENAI_CODEX_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": OPENAI_CODEX_SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "pi",
    }
    return f"{OPENAI_CODEX_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def _capture_oauth_callback_code(*, expected_state: str, timeout_seconds: int, redirect_uri: str) -> str | None:
    parsed = urllib.parse.urlparse(redirect_uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 80

    server = _OAuthHTTPServer((host, port), _OAuthCallbackHandler)
    server.callback_query = None

    event = threading.Event()

    def _serve_one() -> None:
        try:
            server.handle_request()
        finally:
            event.set()
            server.server_close()

    thread = threading.Thread(target=_serve_one, daemon=True)
    thread.start()
    event.wait(timeout_seconds)

    if not server.callback_query:
        return None

    state = server.callback_query.get("state")
    if state != expected_state:
        raise ValueError("OAuth state mismatch. Please retry login.")

    code = server.callback_query.get("code")
    if not code:
        raise ValueError("OAuth callback did not include an authorization code.")
    return code


def _parse_redirect_url_for_code(*, redirect_url: str, expected_state: str) -> str:
    parsed = urllib.parse.urlparse(redirect_url)
    query = urllib.parse.parse_qs(parsed.query)
    state = query.get("state", [None])[0]
    if state != expected_state:
        raise ValueError("OAuth state mismatch in pasted redirect URL. Please retry login.")
    code = query.get("code", [None])[0]
    if not code:
        raise ValueError("Pasted redirect URL does not contain an OAuth code.")
    return code


def _exchange_openai_codex_oauth_token(*, code: str, code_verifier: str, redirect_uri: str) -> dict[str, Any]:
    payload = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": OPENAI_CODEX_CLIENT_ID,
            "code_verifier": code_verifier,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        OPENAI_CODEX_OAUTH_TOKEN_URL,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        body = json.loads(response.read().decode("utf-8"))

    if not isinstance(body, dict):
        raise ValueError("OAuth token response was not a JSON object.")

    refresh_token = body.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise ValueError("OAuth token response is missing refresh_token.")

    return {
        "refresh_token": refresh_token,
        "client_id": OPENAI_CODEX_CLIENT_ID,
    }


def _login_openai_codex_oauth(*, timeout_seconds: int = 120) -> dict[str, Any]:
    code_verifier = _base64url(secrets.token_bytes(64))
    code_challenge = _base64url(hashlib.sha256(code_verifier.encode("utf-8")).digest())
    state = secrets.token_hex(16)

    authorize_url = _build_openai_codex_authorize_url(
        state=state,
        code_challenge=code_challenge,
        redirect_uri=OPENAI_CODEX_REDIRECT_URI,
    )

    print("OpenAI Codex OAuth")
    print("Browser will open for OpenAI authentication.")
    print("If the callback does not auto-complete, paste the redirect URL.")
    print(f"Open: {authorize_url}")

    webbrowser.open(authorize_url)

    code = _capture_oauth_callback_code(
        expected_state=state,
        timeout_seconds=timeout_seconds,
        redirect_uri=OPENAI_CODEX_REDIRECT_URI,
    )
    if code is None:
        redirect_url = input("Paste redirect URL: ").strip()
        code = _parse_redirect_url_for_code(redirect_url=redirect_url, expected_state=state)

    return _exchange_openai_codex_oauth_token(
        code=code,
        code_verifier=code_verifier,
        redirect_uri=OPENAI_CODEX_REDIRECT_URI,
    )


def _cmd_models_auth_login(args: argparse.Namespace) -> int:
    if args.provider != OPENAI_CODEX_PROVIDER:
        raise ValueError(f"Unsupported provider '{args.provider}'. Currently supported: {OPENAI_CODEX_PROVIDER}")

    config_path = AppConfig.resolve_config_path(args.config)
    config_data = _load_yaml(config_path)

    oauth_credential = None
    if args.oauth:
        oauth_credential = _login_openai_codex_oauth()

    _, message = _upsert_openai_codex_model(config_data, set_default=args.set_default, oauth_credential=oauth_credential)
    _save_yaml(config_path, config_data)

    print(message)
    print(f"Config updated: {config_path}")
    if oauth_credential is not None:
        print("OpenAI Codex OAuth token saved to model oauth config.")
    else:
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
    login_parser.add_argument(
        "--oauth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run OpenAI Codex OAuth flow to fetch a refresh token and write model-level oauth config",
    )
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
