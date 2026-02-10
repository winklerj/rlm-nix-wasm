"""Tests for the Wasm sandbox."""

import os

import pytest

from rlm.evaluator.wasm_sandbox import _build_wrapper


# --- Always-run tests (no Wasm needed) ---


class TestBuildWrapper:
    def test_includes_vars_json_loading(self):
        wrapper = _build_wrapper("x = 1")
        assert "/sandbox/vars.json" in wrapper

    def test_includes_result_auto_print(self):
        wrapper = _build_wrapper("result = 'hello'")
        assert "print(result)" in wrapper

    def test_includes_user_code(self):
        wrapper = _build_wrapper("x = 1 + 2\nresult = str(x)")
        assert "x = 1 + 2" in wrapper
        assert "result = str(x)" in wrapper


# --- Wasm-dependent tests ---


needs_wasm = pytest.mark.skipif(
    not os.environ.get("RLM_WASM_PYTHON_PATH"),
    reason="RLM_WASM_PYTHON_PATH not set",
)


@needs_wasm
class TestWasmSandbox:
    def test_available(self, wasm_sandbox):
        assert wasm_sandbox.available

    def test_simple_print(self, wasm_sandbox):
        out = wasm_sandbox.run('print("hello")', {})
        assert "hello" in out

    def test_variable_injection(self, wasm_sandbox):
        out = wasm_sandbox.run("print(len(myvar))", {"myvar": "hello world"})
        assert out.strip() == "11"

    def test_result_auto_print(self, wasm_sandbox):
        out = wasm_sandbox.run("result = str(2 + 3)", {})
        assert out.strip() == "5"

    def test_multiple_variables(self, wasm_sandbox):
        out = wasm_sandbox.run(
            "result = str(int(a) + int(b))", {"a": "10", "b": "20"}
        )
        assert out.strip() == "30"

    def test_re_module(self, wasm_sandbox):
        out = wasm_sandbox.run(
            'import re\nresult = str(len(re.findall(r"\\d+", data)))',
            {"data": "user 123 and user 456"},
        )
        assert out.strip() == "2"

    def test_fuel_exhaustion(self, wasm_sandbox_factory):
        sandbox = wasm_sandbox_factory(fuel=1_000_000)
        with pytest.raises(TimeoutError):
            sandbox.run("while True: pass", {})

    def test_syntax_error(self, wasm_sandbox):
        with pytest.raises(RuntimeError, match="SyntaxError|invalid syntax"):
            wasm_sandbox.run("def def def", {})

    def test_host_filesystem_blocked(self, wasm_sandbox):
        with pytest.raises(RuntimeError):
            wasm_sandbox.run('open("/etc/passwd").read()', {})

    def test_network_blocked(self, wasm_sandbox):
        with pytest.raises(RuntimeError):
            wasm_sandbox.run(
                "import socket; s=socket.socket(); s.connect(('8.8.8.8', 53))",
                {},
            )
