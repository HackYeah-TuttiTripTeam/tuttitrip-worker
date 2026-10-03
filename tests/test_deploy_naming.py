"""Branch -> environment naming of deploy/ (must match the backend's)."""

import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # runs our own bash lib
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parents[1] / "deploy" / "lib.sh"
BASH = shutil.which("bash")


def naming(branch: str) -> list[str]:
    assert BASH is not None
    script = (
        f'. "{LIB}"; e=$(tt_env "$1"); '
        'printf "%s\\n" "$e" "$(tt_image "$e")" "$(tt_container "$e")" '
        '"$(tt_env_file "$e")" "$(tt_api_url "$e")"'
    )
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [BASH, "-c", script, "naming", branch],
        capture_output=True,
        text=True,
        check=True,
        env={"HOME": "/home/u", "PATH": "/usr/bin:/bin"},
    )
    return result.stdout.splitlines()


@pytest.mark.parametrize(
    ("branch", "expected"),
    [
        (
            "main",
            [
                "main",
                "tuttitrip-worker:main",
                "tuttitrip-worker-main",
                "/home/u/tuttitrip/envs/main.worker.env",
                "https://tuttitrip-api.gburek.app",
            ],
        ),
        (
            "develop",
            [
                "develop",
                "tuttitrip-worker:develop",
                "tuttitrip-worker-develop",
                "/home/u/tuttitrip/envs/develop.worker.env",
                "https://tuttitrip-api-develop.gburek.app",
            ],
        ),
        (
            "feature/cos tam",
            [
                "feature-cos-tam",
                "tuttitrip-worker:feature-cos-tam",
                "tuttitrip-worker-feature-cos-tam",
                "/home/u/tuttitrip/envs/feature-cos-tam.worker.env",
                "https://tuttitrip-api-feature-cos-tam.gburek.app",
            ],
        ),
        (
            "Fix/--Weird__Name--",
            [
                "fix-weird-name",
                "tuttitrip-worker:fix-weird-name",
                "tuttitrip-worker-fix-weird-name",
                "/home/u/tuttitrip/envs/fix-weird-name.worker.env",
                "https://tuttitrip-api-fix-weird-name.gburek.app",
            ],
        ),
    ],
)
def test_branch_naming(branch: str, expected: list[str]) -> None:
    assert naming(branch) == expected


def test_slug_is_capped_like_the_backend() -> None:
    env = naming("feature/" + "x" * 100 + "-tail")[0]
    assert len(env) <= 49
    assert not env.endswith("-")
