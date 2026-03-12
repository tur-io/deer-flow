"""Tests for src.models.factory.create_chat_model."""

from __future__ import annotations

import pytest
from langchain.chat_models import BaseChatModel

from src.config.app_config import AppConfig
from src.config.model_config import ModelConfig
from src.config.sandbox_config import SandboxConfig
from src.models import factory as factory_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app_config(models: list[ModelConfig]) -> AppConfig:
    return AppConfig(
        models=models,
        sandbox=SandboxConfig(use="src.sandbox.local:LocalSandboxProvider"),
    )


def _make_model(
    name: str = "test-model",
    *,
    use: str = "langchain_openai:ChatOpenAI",
    supports_thinking: bool = False,
    supports_reasoning_effort: bool = False,
    when_thinking_enabled: dict | None = None,
    thinking: dict | None = None,
) -> ModelConfig:
    return ModelConfig(
        name=name,
        display_name=name,
        description=None,
        use=use,
        model=name,
        supports_thinking=supports_thinking,
        supports_reasoning_effort=supports_reasoning_effort,
        when_thinking_enabled=when_thinking_enabled,
        thinking=thinking,
        supports_vision=False,
    )


class FakeChatModel(BaseChatModel):
    """Minimal BaseChatModel stub that records the kwargs it was called with."""

    captured_kwargs: dict = {}

    def __init__(self, **kwargs):
        # Store kwargs before pydantic processes them
        FakeChatModel.captured_kwargs = dict(kwargs)
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    def _stream(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError


def _patch_factory(monkeypatch, app_config: AppConfig, model_class=FakeChatModel):
    """Patch get_app_config, resolve_class, and tracing for isolated unit tests."""
    monkeypatch.setattr(factory_module, "get_app_config", lambda: app_config)
    monkeypatch.setattr(factory_module, "resolve_class", lambda path, base: model_class)
    monkeypatch.setattr(factory_module, "is_tracing_enabled", lambda: False)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def test_uses_first_model_when_name_is_none(monkeypatch):
    cfg = _make_app_config([_make_model("alpha"), _make_model("beta")])
    _patch_factory(monkeypatch, cfg)

    FakeChatModel.captured_kwargs = {}
    factory_module.create_chat_model(name=None)

    # resolve_class is called — if we reach here without ValueError, the correct model was used
    assert FakeChatModel.captured_kwargs.get("model") == "alpha"


def test_raises_when_model_not_found(monkeypatch):
    cfg = _make_app_config([_make_model("only-model")])
    monkeypatch.setattr(factory_module, "get_app_config", lambda: cfg)
    monkeypatch.setattr(factory_module, "is_tracing_enabled", lambda: False)

    with pytest.raises(ValueError, match="ghost-model"):
        factory_module.create_chat_model(name="ghost-model")


# ---------------------------------------------------------------------------
# thinking_enabled=True
# ---------------------------------------------------------------------------


def test_thinking_enabled_raises_when_not_supported_but_when_thinking_enabled_is_set(monkeypatch):
    """supports_thinking guard fires only when when_thinking_enabled is configured —
    the factory uses that as the signal that the caller explicitly expects thinking to work."""
    wte = {"thinking": {"type": "enabled", "budget_tokens": 5000}}
    cfg = _make_app_config([_make_model("no-think", supports_thinking=False, when_thinking_enabled=wte)])
    _patch_factory(monkeypatch, cfg)

    with pytest.raises(ValueError, match="does not support thinking"):
        factory_module.create_chat_model(name="no-think", thinking_enabled=True)


def test_thinking_enabled_raises_for_empty_when_thinking_enabled_explicitly_set(monkeypatch):
    """supports_thinking guard fires when when_thinking_enabled is set to an empty dict —
    the user explicitly provided the section, so the guard must still fire even though
    effective_wte would be falsy."""
    cfg = _make_app_config([_make_model("no-think-empty", supports_thinking=False, when_thinking_enabled={})])
    _patch_factory(monkeypatch, cfg)

    with pytest.raises(ValueError, match="does not support thinking"):
        factory_module.create_chat_model(name="no-think-empty", thinking_enabled=True)


def test_thinking_enabled_merges_when_thinking_enabled_settings(monkeypatch):
    wte = {"temperature": 1.0, "max_tokens": 16000}
    cfg = _make_app_config([_make_model("thinker", supports_thinking=True, when_thinking_enabled=wte)])
    _patch_factory(monkeypatch, cfg)

    FakeChatModel.captured_kwargs = {}
    factory_module.create_chat_model(name="thinker", thinking_enabled=True)

    assert FakeChatModel.captured_kwargs.get("temperature") == 1.0
    assert FakeChatModel.captured_kwargs.get("max_tokens") == 16000


# ---------------------------------------------------------------------------
# thinking_enabled=False — disable logic
# ---------------------------------------------------------------------------


def test_thinking_disabled_openai_gateway_format(monkeypatch):
    """When thinking is configured via extra_body (OpenAI-compatible gateway),
    disabling must inject extra_body.thinking.type=disabled and reasoning_effort=minimal."""
    wte = {"extra_body": {"thinking": {"type": "enabled", "budget_tokens": 10000}}}
    cfg = _make_app_config(
        [
            _make_model(
                "openai-gw",
                supports_thinking=True,
                supports_reasoning_effort=True,
                when_thinking_enabled=wte,
            )
        ]
    )
    _patch_factory(monkeypatch, cfg)

    captured: dict = {}

    class CapturingModel(FakeChatModel):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            BaseChatModel.__init__(self, **kwargs)

    monkeypatch.setattr(factory_module, "resolve_class", lambda path, base: CapturingModel)

    factory_module.create_chat_model(name="openai-gw", thinking_enabled=False)

    assert captured.get("extra_body") == {"thinking": {"type": "disabled"}}
    assert captured.get("reasoning_effort") == "minimal"
    assert "thinking" not in captured  # must NOT set the direct thinking param


def test_thinking_disabled_langchain_anthropic_format(monkeypatch):
    """When thinking is configured as a direct param (langchain_anthropic),
    disabling must inject thinking.type=disabled WITHOUT touching extra_body or reasoning_effort."""
    wte = {"thinking": {"type": "enabled", "budget_tokens": 8000}}
    cfg = _make_app_config(
        [
            _make_model(
                "anthropic-native",
                use="langchain_anthropic:ChatAnthropic",
                supports_thinking=True,
                supports_reasoning_effort=False,
                when_thinking_enabled=wte,
            )
        ]
    )
    _patch_factory(monkeypatch, cfg)

    captured: dict = {}

    class CapturingModel(FakeChatModel):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            BaseChatModel.__init__(self, **kwargs)

    monkeypatch.setattr(factory_module, "resolve_class", lambda path, base: CapturingModel)

    factory_module.create_chat_model(name="anthropic-native", thinking_enabled=False)

    assert captured.get("thinking") == {"type": "disabled"}
    assert "extra_body" not in captured
    # reasoning_effort must be cleared (supports_reasoning_effort=False)
    assert captured.get("reasoning_effort") is None


def test_thinking_disabled_no_when_thinking_enabled_does_nothing(monkeypatch):
    """If when_thinking_enabled is not set, disabling thinking must not inject any kwargs."""
    cfg = _make_app_config([_make_model("plain", supports_thinking=True, when_thinking_enabled=None)])
    _patch_factory(monkeypatch, cfg)

    captured: dict = {}

    class CapturingModel(FakeChatModel):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            BaseChatModel.__init__(self, **kwargs)

    monkeypatch.setattr(factory_module, "resolve_class", lambda path, base: CapturingModel)

    factory_module.create_chat_model(name="plain", thinking_enabled=False)

    assert "extra_body" not in captured
    assert "thinking" not in captured
    # reasoning_effort not forced (supports_reasoning_effort defaults to False → cleared)
    assert captured.get("reasoning_effort") is None


# ---------------------------------------------------------------------------
# reasoning_effort stripping
# ---------------------------------------------------------------------------


def test_reasoning_effort_cleared_when_not_supported(monkeypatch):
    cfg = _make_app_config([_make_model("no-effort", supports_reasoning_effort=False)])
    _patch_factory(monkeypatch, cfg)

    captured: dict = {}

    class CapturingModel(FakeChatModel):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            BaseChatModel.__init__(self, **kwargs)

    monkeypatch.setattr(factory_module, "resolve_class", lambda path, base: CapturingModel)

    factory_module.create_chat_model(name="no-effort", thinking_enabled=False)

    assert captured.get("reasoning_effort") is None


def test_reasoning_effort_preserved_when_supported(monkeypatch):
    wte = {"extra_body": {"thinking": {"type": "enabled", "budget_tokens": 5000}}}
    cfg = _make_app_config(
        [
            _make_model(
                "effort-model",
                supports_thinking=True,
                supports_reasoning_effort=True,
                when_thinking_enabled=wte,
            )
        ]
    )
    _patch_factory(monkeypatch, cfg)

    captured: dict = {}

    class CapturingModel(FakeChatModel):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            BaseChatModel.__init__(self, **kwargs)

    monkeypatch.setattr(factory_module, "resolve_class", lambda path, base: CapturingModel)

    factory_module.create_chat_model(name="effort-model", thinking_enabled=False)

    # When supports_reasoning_effort=True, it should NOT be cleared to None
    # The disable path sets it to "minimal"; supports_reasoning_effort=True keeps it
    assert captured.get("reasoning_effort") == "minimal"


# ---------------------------------------------------------------------------
# thinking shortcut field
# ---------------------------------------------------------------------------


def test_thinking_shortcut_enables_thinking_when_thinking_enabled(monkeypatch):
    """thinking shortcut alone should act as when_thinking_enabled with a `thinking` key."""
    thinking_settings = {"type": "enabled", "budget_tokens": 8000}
    cfg = _make_app_config(
        [
            _make_model(
                "shortcut-model",
                use="langchain_anthropic:ChatAnthropic",
                supports_thinking=True,
                thinking=thinking_settings,
            )
        ]
    )
    _patch_factory(monkeypatch, cfg)

    captured: dict = {}

    class CapturingModel(FakeChatModel):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            BaseChatModel.__init__(self, **kwargs)

    monkeypatch.setattr(factory_module, "resolve_class", lambda path, base: CapturingModel)

    factory_module.create_chat_model(name="shortcut-model", thinking_enabled=True)

    assert captured.get("thinking") == thinking_settings


def test_thinking_shortcut_disables_thinking_when_thinking_disabled(monkeypatch):
    """thinking shortcut should participate in the disable path (langchain_anthropic format)."""
    thinking_settings = {"type": "enabled", "budget_tokens": 8000}
    cfg = _make_app_config(
        [
            _make_model(
                "shortcut-disable",
                use="langchain_anthropic:ChatAnthropic",
                supports_thinking=True,
                supports_reasoning_effort=False,
                thinking=thinking_settings,
            )
        ]
    )
    _patch_factory(monkeypatch, cfg)

    captured: dict = {}

    class CapturingModel(FakeChatModel):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            BaseChatModel.__init__(self, **kwargs)

    monkeypatch.setattr(factory_module, "resolve_class", lambda path, base: CapturingModel)

    factory_module.create_chat_model(name="shortcut-disable", thinking_enabled=False)

    assert captured.get("thinking") == {"type": "disabled"}
    assert "extra_body" not in captured


def test_thinking_shortcut_merges_with_when_thinking_enabled(monkeypatch):
    """thinking shortcut should be merged into when_thinking_enabled when both are provided."""
    thinking_settings = {"type": "enabled", "budget_tokens": 8000}
    wte = {"max_tokens": 16000}
    cfg = _make_app_config(
        [
            _make_model(
                "merge-model",
                use="langchain_anthropic:ChatAnthropic",
                supports_thinking=True,
                thinking=thinking_settings,
                when_thinking_enabled=wte,
            )
        ]
    )
    _patch_factory(monkeypatch, cfg)

    captured: dict = {}

    class CapturingModel(FakeChatModel):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            BaseChatModel.__init__(self, **kwargs)

    monkeypatch.setattr(factory_module, "resolve_class", lambda path, base: CapturingModel)

    factory_module.create_chat_model(name="merge-model", thinking_enabled=True)

    # Both the thinking shortcut and when_thinking_enabled settings should be applied
    assert captured.get("thinking") == thinking_settings
    assert captured.get("max_tokens") == 16000


def test_thinking_shortcut_not_leaked_into_model_when_disabled(monkeypatch):
    """thinking shortcut must not be passed raw to the model constructor (excluded from model_dump)."""
    thinking_settings = {"type": "enabled", "budget_tokens": 8000}
    cfg = _make_app_config(
        [
            _make_model(
                "no-leak",
                use="langchain_anthropic:ChatAnthropic",
                supports_thinking=True,
                supports_reasoning_effort=False,
                thinking=thinking_settings,
            )
        ]
    )
    _patch_factory(monkeypatch, cfg)

    captured: dict = {}

    class CapturingModel(FakeChatModel):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            BaseChatModel.__init__(self, **kwargs)

    monkeypatch.setattr(factory_module, "resolve_class", lambda path, base: CapturingModel)

    factory_module.create_chat_model(name="no-leak", thinking_enabled=False)

    # The disable path should have set thinking to disabled (not the raw enabled shortcut)
    assert captured.get("thinking") == {"type": "disabled"}


def test_model_oauth_refresh_token_uses_api_key_as_refresh_token(monkeypatch):
    cfg = _make_app_config(
        [
            ModelConfig(
                name="oauth-model",
                display_name="oauth-model",
                description=None,
                use="langchain_openai:ChatOpenAI",
                model="oauth-model",
                api_key="seed-refresh-token",
                oauth={
                    "enabled": True,
                    "token_url": "https://auth.example.com/oauth/token",
                    "grant_type": "refresh_token",
                },
                supports_vision=False,
            )
        ]
    )
    _patch_factory(monkeypatch, cfg)


    captured: dict = {}

    class CapturingModel(FakeChatModel):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            BaseChatModel.__init__(self, **kwargs)

    monkeypatch.setattr(factory_module, "resolve_class", lambda path, base: CapturingModel)

    # Patch injector to avoid async invocation path in this unit test while keeping behavior coverage.

    def fake_inject(model_name, settings, oauth):
        assert model_name == "oauth-model"
        resolved = factory_module._resolve_oauth_model_credentials(settings, oauth)
        assert resolved.refresh_token == "seed-refresh-token"
        updated = dict(settings)
        updated["api_key"] = "fresh-access-token"
        updated["default_headers"] = {"Authorization": "Bearer fresh-access-token"}
        return updated

    monkeypatch.setattr(factory_module, "_inject_model_oauth_credentials", fake_inject)

    factory_module.create_chat_model(name="oauth-model", thinking_enabled=False)

    assert captured.get("api_key") == "fresh-access-token"
    assert captured.get("default_headers") == {"Authorization": "Bearer fresh-access-token"}


def test_resolve_oauth_model_credentials_keeps_explicit_refresh_token():
    settings = {"api_key": "seed-refresh-token"}
    oauth = ModelConfig(
        name="m",
        display_name="m",
        description=None,
        use="langchain_openai:ChatOpenAI",
        model="m",
        supports_vision=False,
        oauth={
            "enabled": True,
            "token_url": "https://auth.example.com/oauth/token",
            "grant_type": "refresh_token",
            "refresh_token": "explicit-refresh",
        },
    ).oauth

    assert oauth is not None
    resolved = factory_module._resolve_oauth_model_credentials(settings, oauth)
    assert resolved.refresh_token == "explicit-refresh"


def test_resolve_oauth_model_credentials_leaves_non_refresh_grants_unchanged():
    settings = {"api_key": "seed-refresh-token"}
    oauth = ModelConfig(
        name="m",
        display_name="m",
        description=None,
        use="langchain_openai:ChatOpenAI",
        model="m",
        supports_vision=False,
        oauth={
            "enabled": True,
            "token_url": "https://auth.example.com/oauth/token",
            "grant_type": "client_credentials",
            "client_id": "id",
            "client_secret": "secret",
        },
    ).oauth

    assert oauth is not None
    resolved = factory_module._resolve_oauth_model_credentials(settings, oauth)
    assert resolved.refresh_token is None



def test_build_oauth_token_request_data_client_credentials():
    oauth = ModelConfig(
        name="m",
        display_name="m",
        description=None,
        use="langchain_openai:ChatOpenAI",
        model="m",
        supports_vision=False,
        oauth={
            "enabled": True,
            "token_url": "https://auth.example.com/oauth/token",
            "grant_type": "client_credentials",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "scope": "scope-a",
            "audience": "aud-1",
            "extra_token_params": {"resource": "codex"},
        },
    ).oauth
    assert oauth is not None

    data = factory_module._build_oauth_token_request_data(oauth)

    assert data == {
        "grant_type": "client_credentials",
        "client_id": "client-id",
        "client_secret": "client-secret",
        "scope": "scope-a",
        "audience": "aud-1",
        "resource": "codex",
    }


def test_build_oauth_token_request_data_refresh_token_requires_refresh_token():
    oauth = ModelConfig(
        name="m",
        display_name="m",
        description=None,
        use="langchain_openai:ChatOpenAI",
        model="m",
        supports_vision=False,
        oauth={
            "enabled": True,
            "token_url": "https://auth.example.com/oauth/token",
            "grant_type": "refresh_token",
        },
    ).oauth
    assert oauth is not None

    with pytest.raises(ValueError, match="requires refresh_token"):
        factory_module._build_oauth_token_request_data(oauth)


def test_fetch_oauth_access_token_parses_custom_fields(monkeypatch):
    oauth = ModelConfig(
        name="m",
        display_name="m",
        description=None,
        use="langchain_openai:ChatOpenAI",
        model="m",
        supports_vision=False,
        oauth={
            "enabled": True,
            "token_url": "https://auth.example.com/oauth/token",
            "grant_type": "refresh_token",
            "refresh_token": "r1",
            "token_field": "token",
            "token_type_field": "typ",
            "default_token_type": "Bearer",
        },
    ).oauth
    assert oauth is not None

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"token": "abc123", "typ": "Bearer"}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr(factory_module.httpx, "Client", lambda timeout: _Client())

    token = factory_module._fetch_oauth_access_token(oauth)
    assert token.token_type == "Bearer"
    assert token.access_token == "abc123"


def test_get_cached_or_fetch_oauth_token_uses_cache_until_expiring(monkeypatch):
    oauth = ModelConfig(
        name="m",
        display_name="m",
        description=None,
        use="langchain_openai:ChatOpenAI",
        model="m",
        supports_vision=False,
        oauth={
            "enabled": True,
            "token_url": "https://auth.example.com/oauth/token",
            "grant_type": "client_credentials",
            "client_id": "id",
            "client_secret": "secret",
            "refresh_skew_seconds": 30,
        },
    ).oauth
    assert oauth is not None

    factory_module._MODEL_OAUTH_TOKEN_CACHE.clear()
    calls = {"n": 0}

    def fake_fetch(_):
        calls["n"] += 1
        access_token = "t" + str(calls["n"])
        return factory_module._ModelOAuthToken(
            token_type="Bearer",
            access_token=access_token,
            expires_at=factory_module.datetime.now(factory_module.UTC) + factory_module.timedelta(seconds=3600),
        )

    monkeypatch.setattr(factory_module, "_fetch_oauth_access_token", fake_fetch)

    a = factory_module._get_cached_or_fetch_oauth_token("model-a", oauth)
    b = factory_module._get_cached_or_fetch_oauth_token("model-a", oauth)

    assert calls["n"] == 1
    assert a.access_token == "t1"
    assert b.access_token == "t1"


def test_get_cached_or_fetch_oauth_token_refreshes_when_expiring(monkeypatch):
    oauth = ModelConfig(
        name="m",
        display_name="m",
        description=None,
        use="langchain_openai:ChatOpenAI",
        model="m",
        supports_vision=False,
        oauth={
            "enabled": True,
            "token_url": "https://auth.example.com/oauth/token",
            "grant_type": "client_credentials",
            "client_id": "id",
            "client_secret": "secret",
            "refresh_skew_seconds": 60,
        },
    ).oauth
    assert oauth is not None

    factory_module._MODEL_OAUTH_TOKEN_CACHE.clear()
    near_expiry = factory_module._ModelOAuthToken(
        token_type="Bearer",
        access_token="old",
        expires_at=factory_module.datetime.now(factory_module.UTC) + factory_module.timedelta(seconds=5),
    )
    cache_key = factory_module._oauth_cache_key("model-a", oauth)
    factory_module._MODEL_OAUTH_TOKEN_CACHE[cache_key] = near_expiry

    monkeypatch.setattr(
        factory_module,
        "_fetch_oauth_access_token",
        lambda _: factory_module._ModelOAuthToken(
            token_type="Bearer",
            access_token="new",
            expires_at=factory_module.datetime.now(factory_module.UTC) + factory_module.timedelta(seconds=3600),
        ),
    )

    token = factory_module._get_cached_or_fetch_oauth_token("model-a", oauth)

    assert token.access_token == "new"


def test_oauth_cache_key_changes_with_oauth_configuration():
    base = ModelConfig(
        name="m",
        display_name="m",
        description=None,
        use="langchain_openai:ChatOpenAI",
        model="m",
        supports_vision=False,
        oauth={
            "enabled": True,
            "token_url": "https://auth.example.com/oauth/token",
            "grant_type": "client_credentials",
            "client_id": "id-1",
            "client_secret": "secret",
        },
    ).oauth
    changed = ModelConfig(
        name="m",
        display_name="m",
        description=None,
        use="langchain_openai:ChatOpenAI",
        model="m",
        supports_vision=False,
        oauth={
            "enabled": True,
            "token_url": "https://auth.example.com/oauth/token",
            "grant_type": "client_credentials",
            "client_id": "id-2",
            "client_secret": "secret",
        },
    ).oauth

    assert base is not None
    assert changed is not None

    key_a = factory_module._oauth_cache_key("model-a", base)
    key_b = factory_module._oauth_cache_key("model-a", changed)

    assert key_a != key_b


def test_fetch_oauth_access_token_supports_json_request_format(monkeypatch):
    oauth = ModelConfig(
        name="m",
        display_name="m",
        description=None,
        use="langchain_openai:ChatOpenAI",
        model="m",
        supports_vision=False,
        oauth={
            "enabled": True,
            "token_url": "https://auth.example.com/oauth/token",
            "grant_type": "refresh_token",
            "refresh_token": "r1",
            "token_request_format": "json",
            "token_request_headers": {"x-test": "1"},
        },
    ).oauth
    assert oauth is not None

    captured_kwargs = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"access_token": "abc123", "token_type": "Bearer"}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            return _Resp()

    monkeypatch.setattr(factory_module.httpx, "Client", lambda timeout: _Client())

    token = factory_module._fetch_oauth_access_token(oauth)

    assert token.access_token == "abc123"
    assert "json" in captured_kwargs
    assert "data" not in captured_kwargs
    assert captured_kwargs.get("headers") == {"x-test": "1"}


def test_oauth_cache_key_changes_with_request_format():
    form_oauth = ModelConfig(
        name="m",
        display_name="m",
        description=None,
        use="langchain_openai:ChatOpenAI",
        model="m",
        supports_vision=False,
        oauth={
            "enabled": True,
            "token_url": "https://auth.example.com/oauth/token",
            "grant_type": "client_credentials",
            "client_id": "id-1",
            "client_secret": "secret",
            "token_request_format": "form",
        },
    ).oauth
    json_oauth = ModelConfig(
        name="m",
        display_name="m",
        description=None,
        use="langchain_openai:ChatOpenAI",
        model="m",
        supports_vision=False,
        oauth={
            "enabled": True,
            "token_url": "https://auth.example.com/oauth/token",
            "grant_type": "client_credentials",
            "client_id": "id-1",
            "client_secret": "secret",
            "token_request_format": "json",
        },
    ).oauth

    assert form_oauth is not None
    assert json_oauth is not None

    form_key = factory_module._oauth_cache_key("model-a", form_oauth)
    json_key = factory_module._oauth_cache_key("model-a", json_oauth)

    assert form_key != json_key
