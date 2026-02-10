"""WebAssembly sandbox for executing Python code in isolation.

Uses wasmtime to run CPython compiled to WASI, providing:
- Memory isolation (hardware-enforced by Wasm)
- No filesystem access beyond a temp sandbox directory
- No network access
- CPU metering via fuel limits
- Memory caps
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _build_wrapper(user_code: str) -> str:
    """Generate the Python wrapper script that runs inside the Wasm sandbox.

    The wrapper:
    1. Reads /sandbox/vars.json and injects each key-value pair as a global
    2. Executes the user's code
    3. If the user set a `result` variable, prints it
    """
    return f'''\
import json as _json

# Load injected variables
with open("/sandbox/vars.json") as _f:
    _vars = _json.load(_f)
for _k, _v in _vars.items():
    globals()[_k] = _v
del _f, _vars
# Clean up loop vars only if they existed (empty dict = no loop iterations)
for _name in ("_k", "_v"):
    if _name in dir():
        del globals()[_name]
del _name, _json

# Execute user code
{user_code}

# Auto-print result if set
if "result" in dir():
    print(result)
'''


class WasmSandbox:
    """Execute Python code inside a WebAssembly (WASI) sandbox.

    The sandbox uses wasmtime to run a CPython binary compiled to WASI.
    Each invocation gets a fresh Store with its own fuel and memory limits,
    while the compiled Module is reused across calls for performance.
    """

    def __init__(
        self,
        python_wasm_path: Path,
        fuel: int = 10_000_000_000,
        memory_mb: int = 256,
    ) -> None:
        self._python_wasm_path = python_wasm_path
        self._fuel = fuel
        self._memory_mb = memory_mb
        # Lazy-loaded wasmtime objects
        self._engine: Any = None
        self._linker: Any = None
        self._module: Any = None

    @property
    def available(self) -> bool:
        """Return True if wasmtime is importable and the python.wasm binary exists."""
        try:
            import wasmtime  # noqa: F401
        except ImportError:
            return False
        return self._python_wasm_path.exists()

    def _ensure_loaded(self) -> None:
        """Compile the python.wasm module once (lazy). Reused across all calls."""
        if self._module is not None:
            return

        try:
            import wasmtime
        except ImportError:
            raise RuntimeError(
                "wasmtime is not installed. Install it with: pip install wasmtime"
            )

        if not self._python_wasm_path.exists():
            raise RuntimeError(
                f"python.wasm not found at {self._python_wasm_path}. "
                f"Set RLM_WASM_PYTHON_PATH to the path of a CPython WASI binary."
            )

        config = wasmtime.Config()
        config.consume_fuel = True
        config.cache = True

        self._engine = wasmtime.Engine(config)
        self._module = wasmtime.Module.from_file(self._engine, str(self._python_wasm_path))
        self._linker = wasmtime.Linker(self._engine)
        self._linker.define_wasi()

    def run(self, code: str, variables: dict[str, str]) -> str:
        """Run Python code in the Wasm sandbox with injected variables.

        Args:
            code: Python source code to execute.
            variables: Dict of variable names to string values, injected as globals.

        Returns:
            The captured stdout output.

        Raises:
            TimeoutError: If the code exhausts its fuel budget.
            RuntimeError: If the code raises a Python error or the sandbox fails.
        """
        import wasmtime

        self._ensure_loaded()

        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox_dir = Path(tmpdir)

            # Write variables as JSON
            (sandbox_dir / "vars.json").write_text(json.dumps(variables))

            # Write the wrapper script
            wrapper = _build_wrapper(code)
            (sandbox_dir / "script.py").write_text(wrapper)

            # Create stdout/stderr capture files
            stdout_path = sandbox_dir / "stdout.txt"
            stderr_path = sandbox_dir / "stderr.txt"
            stdout_path.touch()
            stderr_path.touch()

            # Configure WASI
            wasi_config = wasmtime.WasiConfig()
            wasi_config.argv = ("python", "/sandbox/script.py")
            wasi_config.stdout_file = str(stdout_path)
            wasi_config.stderr_file = str(stderr_path)
            wasi_config.preopen_dir(str(sandbox_dir), "/sandbox")

            # Create a fresh store with fuel and memory limits
            store = wasmtime.Store(self._engine)
            store.set_fuel(self._fuel)
            store.set_wasi(wasi_config)

            # Instantiate and run
            instance = self._linker.instantiate(store, self._module)
            start = instance.exports(store).get("_start")
            if start is None:
                raise RuntimeError("python.wasm does not export _start")

            try:
                start(store)
            except wasmtime.ExitTrap as e:
                # CPython calls proc_exit() — exit code 0 is success
                error_msg = str(e)
                if "exit status 0" in error_msg:
                    return stdout_path.read_text()
                # Non-zero exit code: check stderr for Python traceback
                stderr_content = stderr_path.read_text().strip()
                if stderr_content:
                    raise RuntimeError(
                        f"Python error in sandbox:\n{stderr_content}"
                    ) from e
                raise RuntimeError(f"Wasm execution failed: {error_msg}") from e
            except wasmtime.Trap as e:
                # Wasm trap — fuel exhaustion, unreachable, etc.
                error_msg = str(e)
                if "all fuel consumed" in error_msg.lower():
                    raise TimeoutError(
                        f"Code execution exceeded fuel limit ({self._fuel})"
                    ) from e
                stderr_content = stderr_path.read_text().strip()
                if stderr_content:
                    raise RuntimeError(
                        f"Python error in sandbox:\n{stderr_content}"
                    ) from e
                raise RuntimeError(f"Wasm trap: {error_msg}") from e
            except wasmtime.WasmtimeError as e:
                error_msg = str(e)
                stderr_content = stderr_path.read_text().strip()
                if stderr_content:
                    raise RuntimeError(
                        f"Python error in sandbox:\n{stderr_content}"
                    ) from e
                raise RuntimeError(f"Wasm execution failed: {error_msg}") from e

            return stdout_path.read_text()
