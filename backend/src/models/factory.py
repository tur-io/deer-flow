import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Lock

import httpx
from langchain.chat_models import BaseChatModel

from src.config import get_app_config, get_tracing_config, is_tracing_enabled
from src.config.model_config import ModelOAuthConfig
from src.reflection import resolve_class

logger = logging.getLogger(__name__)


@dataclass
class _ModelOAuthToken:
    token_type: str
    access_token: str
    expires_at: datetime


_MODEL_OAUTH_TOKEN_CACHE: dict[str, _ModelOAuthToken] = {}
_MODEL_OAUTH_CACHE_LOCK = Lock()


def _resolve_oauth_model_credentials(model_settings: dict, oauth: ModelOAuthConfig) -> ModelOAuthConfig:
    """Allow model-level oauth to inherit credentials from model settings.

    For `refresh_token` grant, if `refresh_token` isn't explicitly configured,
    we reuse `api_key` from model settings as the refresh token source.
    """
    if oauth.grant_type != "refresh_token" or oauth.refresh_token:
        return oauth

    api_key = model_settings.get("api_key")
    if not api_key:
        return oauth

    return oauth.model_copy(update={"refresh_token": api_key})


def _build_oauth_token_request_data(oauth: ModelOAuthConfig) -> dict[str, str]:
    data: dict[str, str] = {
        "grant_type": oauth.grant_type,
        **oauth.extra_token_params,
    }

    if oauth.scope:
        data["scope"] = oauth.scope
    if oauth.audience:
        data["audience"] = oauth.audience

    if oauth.grant_type == "client_credentials":
        if not oauth.client_id or not oauth.client_secret:
            raise ValueError("OAuth client_credentials grant requires client_id and client_secret")
        data["client_id"] = oauth.client_id
        data["client_secret"] = oauth.client_secret
    elif oauth.grant_type == "refresh_token":
        if not oauth.refresh_token:
            raise ValueError("OAuth refresh_token grant requires refresh_token")
        data["refresh_token"] = oauth.refresh_token
        if oauth.client_id:
            data["client_id"] = oauth.client_id
        if oauth.client_secret:
            data["client_secret"] = oauth.client_secret
    else:
        raise ValueError(f"Unsupported OAuth grant type: {oauth.grant_type}")

    return data


def _parse_oauth_token_payload(payload: dict, oauth: ModelOAuthConfig) -> _ModelOAuthToken:
    access_token_raw = payload.get(oauth.token_field)
    if not access_token_raw:
        raise ValueError(f"OAuth token response missing '{oauth.token_field}'")

    token_type = str(payload.get(oauth.token_type_field, oauth.default_token_type) or oauth.default_token_type)

    expires_in_raw = payload.get(oauth.expires_in_field, 3600)
    try:
        expires_in = int(expires_in_raw)
    except (TypeError, ValueError):
        expires_in = 3600

    expires_at = datetime.now(UTC) + timedelta(seconds=max(expires_in, 0))
    return _ModelOAuthToken(token_type=token_type, access_token=str(access_token_raw), expires_at=expires_at)


def _fetch_oauth_access_token(oauth: ModelOAuthConfig) -> _ModelOAuthToken:
    data = _build_oauth_token_request_data(oauth)

    with httpx.Client(timeout=30) as client:
        response = client.post(oauth.token_url, data=data)
        response.raise_for_status()
        payload = response.json()

    return _parse_oauth_token_payload(payload, oauth)


def _is_token_expiring(token: _ModelOAuthToken, oauth: ModelOAuthConfig) -> bool:
    now = datetime.now(UTC)
    return token.expires_at <= now + timedelta(seconds=max(oauth.refresh_skew_seconds, 0))


def _get_cached_or_fetch_oauth_token(model_name: str, oauth: ModelOAuthConfig) -> _ModelOAuthToken:
    with _MODEL_OAUTH_CACHE_LOCK:
        cached = _MODEL_OAUTH_TOKEN_CACHE.get(model_name)
        if cached and not _is_token_expiring(cached, oauth):
            return cached

        fresh = _fetch_oauth_access_token(oauth)
        _MODEL_OAUTH_TOKEN_CACHE[model_name] = fresh
        return fresh


def _inject_model_oauth_credentials(model_name: str, model_settings: dict, oauth: ModelOAuthConfig | None) -> dict:
    if oauth is None or not oauth.enabled:
        return model_settings

    resolved_oauth = _resolve_oauth_model_credentials(model_settings, oauth)
    token = _get_cached_or_fetch_oauth_token(model_name, resolved_oauth)

    updated = dict(model_settings)
    updated["api_key"] = token.access_token

    headers = dict(updated.get("default_headers") or {})
    headers["Authorization"] = f"{token.token_type} {token.access_token}".strip()
    updated["default_headers"] = headers

    return updated


def create_chat_model(name: str | None = None, thinking_enabled: bool = False, **kwargs) -> BaseChatModel:
    """Create a chat model instance from the config.

    Args:
        name: The name of the model to create. If None, the first model in the config will be used.

    Returns:
        A chat model instance.
    """
    config = get_app_config()
    if name is None:
        name = config.models[0].name
    model_config = config.get_model_config(name)
    if model_config is None:
        raise ValueError(f"Model {name} not found in config") from None
    model_class = resolve_class(model_config.use, BaseChatModel)
    model_settings_from_config = model_config.model_dump(
        exclude_none=True,
        exclude={
            "use",
            "name",
            "display_name",
            "description",
            "supports_thinking",
            "supports_reasoning_effort",
            "when_thinking_enabled",
            "thinking",
            "supports_vision",
            "oauth",
        },
    )
    model_settings_from_config = _inject_model_oauth_credentials(name, model_settings_from_config, model_config.oauth)
    # Compute effective when_thinking_enabled by merging in the `thinking` shortcut field.
    # The `thinking` shortcut is equivalent to setting when_thinking_enabled["thinking"].
    has_thinking_settings = (model_config.when_thinking_enabled is not None) or (model_config.thinking is not None)
    effective_wte: dict = dict(model_config.when_thinking_enabled) if model_config.when_thinking_enabled else {}
    if model_config.thinking is not None:
        merged_thinking = {**(effective_wte.get("thinking") or {}), **model_config.thinking}
        effective_wte = {**effective_wte, "thinking": merged_thinking}
    if thinking_enabled and has_thinking_settings:
        if not model_config.supports_thinking:
            raise ValueError(f"Model {name} does not support thinking. Set `supports_thinking` to true in the `config.yaml` to enable thinking.") from None
        if effective_wte:
            model_settings_from_config.update(effective_wte)
    if not thinking_enabled and has_thinking_settings:
        if effective_wte.get("extra_body", {}).get("thinking", {}).get("type"):
            # OpenAI-compatible gateway: thinking is nested under extra_body
            kwargs.update({"extra_body": {"thinking": {"type": "disabled"}}})
            kwargs.update({"reasoning_effort": "minimal"})
        elif effective_wte.get("thinking", {}).get("type"):
            # Native langchain_anthropic: thinking is a direct constructor parameter
            kwargs.update({"thinking": {"type": "disabled"}})
    if not model_config.supports_reasoning_effort and "reasoning_effort" in kwargs:
        del kwargs["reasoning_effort"]

    model_instance = model_class(**kwargs, **model_settings_from_config)

    if is_tracing_enabled():
        try:
            from langchain_core.tracers.langchain import LangChainTracer

            tracing_config = get_tracing_config()
            tracer = LangChainTracer(
                project_name=tracing_config.project,
            )
            existing_callbacks = model_instance.callbacks or []
            model_instance.callbacks = [*existing_callbacks, tracer]
            logger.debug(f"LangSmith tracing attached to model '{name}' (project='{tracing_config.project}')")
        except Exception as e:
            logger.warning(f"Failed to attach LangSmith tracing to model '{name}': {e}")
    return model_instance
