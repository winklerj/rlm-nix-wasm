"""Bubblewrap-based sandboxing for explore operations."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class BwrapSandbox:
    """Execute operations inside a bubblewrap sandbox."""

    def __init__(self) -> None:
        self.bwrap_path = shutil.which("bwrap")

    @property
    def available(self) -> bool:
        return self.bwrap_path is not None

    def run(
        self,
        command: list[str],
        input_data: str | None = None,
        timeout: int = 30,
    ) -> str:
        """Run a command in a sandboxed environment.

        The sandbox has:
        - Read-only access to /usr, /lib, /bin
        - No network access
        - A tmpfs for /tmp
        - Input data available as /sandbox/input if provided
        """
        if not self.available:
            raise RuntimeError("bubblewrap (bwrap) is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox_dir = Path(tmpdir)
            if input_data is not None:
                (sandbox_dir / "input").write_text(input_data)

            assert self.bwrap_path is not None
            bwrap_args: list[str] = [
                self.bwrap_path,
                "--ro-bind", "/usr", "/usr",
                "--ro-bind", "/lib", "/lib",
                "--ro-bind", "/bin", "/bin",
                "--symlink", "/usr/lib64", "/lib64",
                "--proc", "/proc",
                "--dev", "/dev",
                "--tmpfs", "/tmp",
                "--ro-bind", str(sandbox_dir), "/sandbox",
                "--unshare-net",
                "--new-session",
                "--die-with-parent",
                "--", *command,
            ]

            result = subprocess.run(
                bwrap_args,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Sandbox command failed (exit {result.returncode}): {result.stderr}"
                )
            stdout: str = result.stdout
            return stdout
