"""Liveness file used by the Docker HEALTHCHECK."""

from pathlib import Path

from tuttitrip_worker.healthcheck import MAX_AGE_SEC, is_alive


def test_missing_file_is_unhealthy(tmp_path: Path) -> None:
    assert not is_alive(tmp_path / "missing")


def test_fresh_file_is_healthy_and_stale_is_not(tmp_path: Path) -> None:
    path = tmp_path / "alive"
    path.touch()
    mtime = path.stat().st_mtime
    assert is_alive(path, now=mtime + 1)
    assert not is_alive(path, now=mtime + MAX_AGE_SEC + 1)
