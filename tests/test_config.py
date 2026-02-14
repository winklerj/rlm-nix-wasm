"""Tests for configuration loading."""

from rlm.config import load_config


class TestLoadConfig:
    def test_default_model(self):
        config = load_config()
        assert config.model == "claude-opus-4-5"
        assert config.child_model is None

    def test_model_override(self):
        config = load_config(model="gpt-5-nano")
        assert config.model == "gpt-5-nano"

    def test_child_model_override(self):
        config = load_config(child_model="gpt-5-nano")
        assert config.child_model == "gpt-5-nano"

    def test_child_model_env_var(self, monkeypatch):
        monkeypatch.setenv("RLM_CHILD_MODEL", "gpt-5-nano")
        config = load_config()
        assert config.child_model == "gpt-5-nano"

    def test_cli_overrides_env(self, monkeypatch):
        monkeypatch.setenv("RLM_CHILD_MODEL", "env-model")
        config = load_config(child_model="cli-model")
        assert config.child_model == "cli-model"

    def test_none_override_uses_env(self, monkeypatch):
        monkeypatch.setenv("RLM_CHILD_MODEL", "env-model")
        config = load_config(child_model=None)
        assert config.child_model == "env-model"
