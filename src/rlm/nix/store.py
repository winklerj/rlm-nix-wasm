"""Wrapper around nix-store commands."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class NixStoreError(Exception):
    """Error interacting with the Nix store."""
    pass


class NixStore:
    """Wrapper around nix-store for importing/querying store paths."""

    def __init__(self) -> None:
        self.nix_store_path = shutil.which("nix-store")
        self.nix_build_path = shutil.which("nix-build")

    @property
    def available(self) -> bool:
        """Check if Nix is available on this system."""
        return self.nix_store_path is not None and self.nix_build_path is not None

    def add_to_store(self, content: str, name: str = "rlm-input") -> str:
        """Import a string as a file into the Nix store. Returns the store path."""
        if not self.available:
            raise NixStoreError("Nix is not installed")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix=f"{name}-", delete=False
        ) as f:
            f.write(content)
            f.flush()
            tmp_path = f.name

        try:
            result = subprocess.run(
                [self.nix_store_path, "--add", tmp_path],  # type: ignore[list-item]
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise NixStoreError(f"nix-store --add failed: {result.stderr}")
            stdout: str = result.stdout
            return stdout.strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def query_hash(self, store_path: str) -> str:
        """Get the hash of a store path."""
        if not self.available:
            raise NixStoreError("Nix is not installed")

        result = subprocess.run(
            [self.nix_store_path, "--query", "--hash", store_path],  # type: ignore[list-item]
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise NixStoreError(f"nix-store --query --hash failed: {result.stderr}")
        stdout: str = result.stdout
        return stdout.strip()

    def gc(self) -> str:
        """Run garbage collection. Returns the output."""
        if not self.available:
            raise NixStoreError("Nix is not installed")

        result = subprocess.run(
            [self.nix_store_path, "--gc"],  # type: ignore[list-item]
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise NixStoreError(f"nix-store --gc failed: {result.stderr}")
        stdout: str = result.stdout
        return stdout.strip()
