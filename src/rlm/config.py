"""Configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rlm.types import RLMConfig


def load_config(**overrides: Any) -> RLMConfig:
    """Load configuration from environment variables and overrides."""
    env_mappings: dict[str, str | tuple[str, Any]] = {
        "RLM_MODEL": "model",
        "RLM_CHILD_MODEL": "child_model",
        "RLM_MAX_EXPLORE_STEPS": ("max_explore_steps", int),
        "RLM_MAX_COMMIT_CYCLES": ("max_commit_cycles", int),
        "RLM_MAX_RECURSION_DEPTH": ("max_recursion_depth", int),
        "RLM_MAX_PARALLEL_JOBS": ("max_parallel_jobs", int),
        "RLM_CACHE_DIR": ("cache_dir", Path),
        "RLM_USE_NIX": ("use_nix", lambda x: x.lower() in ("1", "true", "yes")),
        "RLM_VERBOSE": ("verbose", lambda x: x.lower() in ("1", "true", "yes")),
    }

    config_data: dict[str, Any] = {}
    for env_var, mapping in env_mappings.items():
        val = os.environ.get(env_var)
        if val is not None:
            if isinstance(mapping, str):
                config_data[mapping] = val
            else:
                field_name, converter = mapping
                config_data[field_name] = converter(val)

    config_data.update({k: v for k, v in overrides.items() if v is not None})
    return RLMConfig(**config_data)
