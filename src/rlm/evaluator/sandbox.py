"""Bubblewrap-based sandboxing for explore operations."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class BwrapSandbox:
    """Execute operations inside a bubblewrap sandbox."""

    def __init__(self) -> None:
        self.bwrap_path = shutil.which("bwrap")
        self._net_unshare_supported: bool | None = None

    @property
    def available(self) -> bool:
        return self.bwrap_path is not None

    def _check_net_unshare(self) -> bool:
        """Probe whether --unshare-net works in this environment."""
        if self._net_unshare_supported is not None:
            return self._net_unshare_supported

        assert self.bwrap_path is not None
        try:
            result = subprocess.run(
                [
                    self.bwrap_path,
                    "--ro-bind", "/usr", "/usr",
                    "--ro-bind", "/lib", "/lib",
                    "--ro-bind", "/bin", "/bin",
                    "--symlink", "/usr/lib64", "/lib64",
                    "--proc", "/proc",
                    "--dev", "/dev",
                    "--tmpfs", "/tmp",
                    "--unshare-net",
                    "--new-session",
                    "--die-with-parent",
                    "--", "true",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self._net_unshare_supported = result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            self._net_unshare_supported = False

        if not self._net_unshare_supported:
            logger.warning(
                "bwrap --unshare-net is not supported in this environment "
                "(common inside containers). Network isolation will be skipped."
            )
        return self._net_unshare_supported

    def run(
        self,
        command: list[str],
        input_data: str | None = None,
        timeout: int = 30,
    ) -> str:
        """Run a command in a sandboxed environment.

        The sandbox has:
        - Read-only access to /usr, /lib, /bin
        - No network access (when the kernel supports it)
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
            ]

            if self._check_net_unshare():
                bwrap_args.append("--unshare-net")

            bwrap_args.extend([
                "--new-session",
                "--die-with-parent",
                "--", *command,
            ])

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
