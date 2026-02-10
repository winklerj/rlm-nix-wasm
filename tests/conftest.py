"""Shared test fixtures."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def wasm_sandbox():
    """Provide a WasmSandbox instance for tests."""
    path = os.environ.get("RLM_WASM_PYTHON_PATH")
    if not path:
        pytest.skip("RLM_WASM_PYTHON_PATH not set")
    from rlm.evaluator.wasm_sandbox import WasmSandbox
    return WasmSandbox(python_wasm_path=Path(path))


@pytest.fixture
def wasm_sandbox_factory():
    """Provide a WasmSandbox factory with configurable fuel."""
    path = os.environ.get("RLM_WASM_PYTHON_PATH")
    if not path:
        pytest.skip("RLM_WASM_PYTHON_PATH not set")
    from rlm.evaluator.wasm_sandbox import WasmSandbox

    def _factory(fuel: int = 10_000_000_000, memory_mb: int = 256) -> WasmSandbox:
        return WasmSandbox(
            python_wasm_path=Path(path), fuel=fuel, memory_mb=memory_mb,
        )
    return _factory
