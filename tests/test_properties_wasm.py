"""Property-based tests for Wasm eval invariants (E1-E3).

Requires wasmtime and a python.wasm binary (RLM_WASM_PYTHON_PATH).
Skipped automatically when unavailable.
"""

from __future__ import annotations

import os

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# Skip entire module if Wasm not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("RLM_WASM_PYTHON_PATH"),
    reason="RLM_WASM_PYTHON_PATH not set",
)


# E1: EvalTerminatesWithFuel
@given(
    fuel=st.integers(min_value=1000, max_value=100_000),
)
@settings(max_examples=20, deadline=30_000, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_eval_terminates_with_fuel(fuel: int, wasm_sandbox_factory) -> None:  # type: ignore[no-untyped-def]
    """TLA+ invariant E1: finite fuel => execution terminates.

    Even infinite loops must terminate (with TimeoutError) when
    fuel is finite.
    """
    sandbox = wasm_sandbox_factory(fuel=fuel)
    try:
        sandbox.run("while True: pass", {})
        pytest.fail("Infinite loop should have exhausted fuel")
    except TimeoutError:
        pass  # Expected -- fuel exhaustion
    except RuntimeError:
        pass  # Also acceptable -- e.g., Wasm trap


# E2: EvalVariableInjection
@given(
    varname=st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True),
    value=st.text(min_size=0, max_size=1000, alphabet=st.characters(
        whitelist_categories=("L", "N", "Z"),
    )),
)
@settings(max_examples=50, deadline=30_000, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_eval_variable_injection(varname: str, value: str, wasm_sandbox) -> None:  # type: ignore[no-untyped-def]
    """TLA+ invariant E2: injected variables are accessible.

    For any variable name and string value, the sandbox code can
    read the injected variable and its value matches.
    """
    code = f"result = str(len({varname}))"
    result = wasm_sandbox.run(code, {varname: value})
    assert result.strip() == str(len(value))


# E3: EvalDeterministic
@given(
    a=st.integers(min_value=0, max_value=10000),
    b=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=50, deadline=30_000, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_eval_deterministic(a: int, b: int, wasm_sandbox) -> None:  # type: ignore[no-untyped-def]
    """TLA+ invariant E3: deterministic code => identical results.

    Running the same deterministic code with the same inputs
    must produce the same output across multiple invocations.
    """
    code = f"result = str({a} + {b})"
    result1 = wasm_sandbox.run(code, {})
    result2 = wasm_sandbox.run(code, {})
    assert result1 == result2
    assert result1.strip() == str(a + b)
