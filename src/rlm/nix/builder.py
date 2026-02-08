"""Nix build wrapper for executing derivations."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from rlm.nix.store import NixStore, NixStoreError


class NixBuilder:
    """Build Nix derivations and retrieve results."""

    def __init__(self, max_jobs: int = 4):
        self.nix_store = NixStore()
        self.max_jobs = max_jobs

    @property
    def available(self) -> bool:
        return self.nix_store.available

    def build(self, nix_expr: str) -> str:
        """Build a Nix expression and return the output store path."""
        if not self.available:
            raise NixStoreError("Nix is not installed")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".nix", prefix="rlm-", delete=False
        ) as f:
            f.write(nix_expr)
            f.flush()
            nix_file = f.name

        try:
            result = subprocess.run(
                [
                    self.nix_store.nix_build_path,  # type: ignore[list-item]
                    nix_file,
                    "--no-out-link",
                    "--max-jobs", str(self.max_jobs),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise NixStoreError(f"nix-build failed: {result.stderr}")
            stdout: str = result.stdout
            return stdout.strip()
        finally:
            Path(nix_file).unlink(missing_ok=True)

    def build_and_read(self, nix_expr: str) -> str:
        """Build a Nix expression and read the output as text."""
        output_path = self.build(nix_expr)
        return Path(output_path).read_text()
