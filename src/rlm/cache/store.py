"""Content-addressed cache store."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from rlm.types import OpType


class CacheStore:
    """Content-addressed file-system cache for operation results.

    Each result is stored at: {cache_dir}/{hash[:2]}/{hash[2:4]}/{hash}
    This mirrors Nix store path structure for familiarity.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        """Get the filesystem path for a cache key."""
        return self.cache_dir / key[:2] / key[2:4] / key

    def get(self, key: str) -> str | None:
        """Look up a cached result. Returns None on miss."""
        path = self._path_for_key(key)
        if path.exists():
            return path.read_text()
        return None

    def put(self, key: str, value: str) -> None:
        """Store a result in the cache."""
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return self._path_for_key(key).exists()

    def stats(self) -> dict[str, object]:
        """Return cache statistics."""
        total_files = 0
        total_bytes = 0
        for p in self.cache_dir.rglob("*"):
            if p.is_file():
                total_files += 1
                total_bytes += p.stat().st_size
        return {
            "entries": total_files,
            "size_bytes": total_bytes,
            "size_human": _human_size(total_bytes),
            "cache_dir": str(self.cache_dir),
        }

    def clear(self) -> int:
        """Clear all cached entries. Returns number of entries removed."""
        current_stats = self.stats()
        entries = current_stats["entries"]
        assert isinstance(entries, int)
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        return entries


def make_cache_key(op: OpType, args: dict, input_hashes: dict[str, str]) -> str:  # type: ignore[type-arg]
    """Compute a deterministic cache key for an operation."""
    resolved_args: dict[str, object] = {}
    for k, v in args.items():
        if isinstance(v, str) and v in input_hashes:
            resolved_args[k] = input_hashes[v]
        else:
            resolved_args[k] = v
    key_data = {"op": op.value, "args": resolved_args}
    return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
