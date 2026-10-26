"""
osm_cache.py

Persistent application-level cache for parsed OSM graphs and GeoDataFrames.

While OSMnx caches raw HTTP JSON responses from Overpass, parsing large JSON
payloads into NetworkX graphs or GeoPandas GeoDataFrames still consumes
substantial CPU time. OSMDataCache serializes the constructed Python objects
to disk so subsequent styling runs load almost instantaneously.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any


logger = logging.getLogger("GenericPlotter")


def _normalize_for_hash(obj: Any) -> Any:
    """
    Recursively normalize Python data structures into deterministic,
    JSON-serializable primitives for stable SHA-256 hashing.
    """
    if isinstance(obj, dict):
        return {
            str(k): _normalize_for_hash(v)
            for k, v in sorted(obj.items(), key=lambda item: str(item[0]))
        }
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_hash(x) for x in obj]
    if isinstance(obj, set):
        return sorted(_normalize_for_hash(x) for x in obj)
    if isinstance(obj, float):
        return round(obj, 8)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


class OSMDataCache:
    """
    Two-tier (in-memory + disk) cache for OSMnx graphs and GeoDataFrames.

    Parameters
    ----------
    cache_dir:
        Directory path for storing pickled cache files. Defaults to ``.osm_cache``.
    ttl_days:
        Time-to-live in days before a disk cache entry is considered expired.
    enabled:
        Master switch to enable or bypass caching.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        ttl_days: int | float = 30,
        enabled: bool = True,
    ):
        self.enabled = bool(enabled)
        self.ttl_seconds = float(ttl_days) * 86400.0
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".osm_cache")

        # In-memory L1 cache for within-process reuse
        self._memory_cache: dict[tuple[str, str], Any] = {}

        if self.enabled:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning(
                    "Could not create cache directory '%s': %s. Disabling disk cache.",
                    self.cache_dir,
                    exc,
                )
                self.enabled = False

    def _make_key(self, namespace: str, payload: dict[str, Any]) -> str:
        """Compute a deterministic SHA-256 digest for the given request payload."""
        normalized = _normalize_for_hash(payload)
        canonical_json = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        digest = hashlib.sha256(
            f"{namespace}:{canonical_json}".encode("utf-8")
        ).hexdigest()
        return digest

    def _get_cache_path(self, namespace: str, key: str) -> Path:
        """Return the file path for a specific namespace and hash key."""
        safe_ns = "".join(c if c.isalnum() or c in "_-" else "_" for c in namespace)
        return self.cache_dir / safe_ns / f"{key}.pkl"

    def get(self, namespace: str, payload: dict[str, Any]) -> Any | None:
        """
        Retrieve an object from the cache.
        Returns None on cache miss, expiration, or corrupted file.
        """
        if not self.enabled:
            return None

        key = self._make_key(namespace, payload)
        mem_key = (namespace, key)

        # 1. Check L1 in-memory cache
        if mem_key in self._memory_cache:
            logger.debug(
                "[OSMDataCache] Memory hit for '%s' (%s)",
                namespace,
                key[:10],
            )
            return self._memory_cache[mem_key]

        # 2. Check L2 disk cache
        path = self._get_cache_path(namespace, key)
        if not path.exists():
            return None

        try:
            if self.ttl_seconds > 0:
                age_seconds = time.time() - path.stat().st_mtime
                if age_seconds > self.ttl_seconds:
                    logger.info(
                        "[OSMDataCache] Expired cache entry for '%s' (%s). Removing.",
                        namespace,
                        key[:10],
                    )
                    path.unlink(missing_ok=True)
                    return None

            with path.open("rb") as f:
                data = pickle.load(f)

            self._memory_cache[mem_key] = data
            logger.info(
                "[OSMDataCache] Disk hit for '%s' (%s)",
                namespace,
                key[:10],
            )
            return data

        except Exception as exc:
            logger.warning(
                "[OSMDataCache] Failed to load cache file '%s' (%s). Deleting corrupted file.",
                path,
                exc,
            )
            path.unlink(missing_ok=True)
            return None

    def set(self, namespace: str, payload: dict[str, Any], value: Any) -> None:
        """Store an object in both memory and disk caches using atomic file replacement."""
        if not self.enabled or value is None:
            return

        key = self._make_key(namespace, payload)
        mem_key = (namespace, key)
        self._memory_cache[mem_key] = value

        path = self._get_cache_path(namespace, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".pkl.tmp")

            with tmp_path.open("wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

            tmp_path.replace(path)
            logger.debug(
                "[OSMDataCache] Saved '%s' (%s) to disk.",
                namespace,
                key[:10],
            )
        except Exception as exc:
            logger.warning(
                "[OSMDataCache] Failed to write cache file '%s': %s",
                path,
                exc,
            )

    def clear(self, namespace: str | None = None) -> int:
        """
        Clear cached entries from memory and disk.
        If ``namespace`` is provided, only that namespace is cleared.
        Returns the number of disk files removed.
        """
        removed_count = 0

        if namespace is None:
            self._memory_cache.clear()
            target_dir = self.cache_dir
            pattern = "**/*.pkl"
        else:
            self._memory_cache = {
                k: v for k, v in self._memory_cache.items() if k[0] != namespace
            }
            safe_ns = "".join(c if c.isalnum() or c in "_-" else "_" for c in namespace)
            target_dir = self.cache_dir / safe_ns
            pattern = "*.pkl"

        if target_dir.exists():
            for pkl_file in target_dir.glob(pattern):
                try:
                    pkl_file.unlink()
                    removed_count += 1
                except OSError:
                    pass

        logger.info("[OSMDataCache] Cleared %d disk cache files.", removed_count)
        return removed_count