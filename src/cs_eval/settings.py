"""Configuration loading and precedence.

Resolution order, highest wins:

    1. explicit CLI flag        --region us-west-2
    2. exported environment     AWS_REGION=us-west-2 make validate
    3. .env file at repo root   AWS_REGION=us-west-2
    4. built-in default         see DEFAULTS below

Exported variables beat `.env` (that is `load_dotenv(override=False)`), so a
one-off override on the command line does not require editing the file.

`load_env()` must run before any boto3 client is constructed, because `AWS_PROFILE`
and `AWS_REGION` are read out of `os.environ` by boto3 at client-creation time.
`cli.main()` calls it as its first statement.

Nothing here requires `.env` to exist. Replay mode works with no `.env` and no
credentials at all.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"

# Built-in defaults: the bottom of the precedence chain.
DEFAULTS: dict[str, str] = {
    # Empty means "use the normal credential chain" (instance role, SSO, env keys).
    # Set a named profile in .env if you use one; no profile name is committed here.
    "AWS_PROFILE": "",
    "AWS_REGION": "us-east-1",
    "CS_EVAL_MODEL_ID": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "CS_EVAL_TEMPERATURE": "0.0",
    "CS_EVAL_MAX_USD": "5.00",
    "CS_EVAL_EVALUATORS": "core",
    "CS_EVAL_DATA": "data/conversations.yaml",
    "CS_EVAL_BASELINE": "baseline",
    "CS_EVAL_IMPROVED": "improved",
    "CS_EVAL_LIMIT": "0",
    "CS_EVAL_MODE": "live",
}


# Keys whose value came from the .env file rather than an exported variable.
_FROM_FILE: set[str] = set()


def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal .env parser, used only if python-dotenv is unavailable.

    Keeps the free `make replay` path working in a stale virtualenv rather than
    crashing on a missing import.
    """
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, separator, value = line.partition("=")
        if not separator:
            continue
        value = value.split(" #", 1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_env(path: Path | None = None) -> Path | None:
    """Load `.env` into os.environ without clobbering exported variables.

    Returns the file that was loaded, or None when there was none.
    """
    env_path = path or ENV_FILE
    if not env_path.exists():
        return None

    file_values = _parse_env_file(env_path)
    # Record provenance before mutating os.environ, so `validate` can tell an
    # exported variable apart from a .env one. Without this, a shell that exports
    # AWS_REGION silently overrides the file and the report looks identical.
    _FROM_FILE.clear()
    _FROM_FILE.update(
        {key for key in file_values if not os.environ.get(key)}
    )

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        for key, value in file_values.items():
            os.environ.setdefault(key, value)

    # An empty AWS_PROFILE is worse than an unset one: boto3 treats it as a
    # profile named "" and fails to resolve credentials. `.env.example` ships
    # AWS_PROFILE= blank on purpose, so strip it rather than pass it through.
    if os.environ.get("AWS_PROFILE", "").strip() == "":
        os.environ.pop("AWS_PROFILE", None)

    # boto3 reads AWS_DEFAULT_REGION in some code paths and AWS_REGION in others;
    # mirror whichever the user set so both agree.
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region:
        os.environ.setdefault("AWS_REGION", region)
        os.environ.setdefault("AWS_DEFAULT_REGION", region)

    return env_path


def get(name: str) -> str:
    """Resolved value for a known setting: env (incl. .env) else built-in default."""
    if name not in DEFAULTS:
        raise KeyError(f"unknown setting: {name}")
    return os.environ.get(name) or DEFAULTS[name]


def get_float(name: str) -> float:
    raw = get(name)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def get_int(name: str) -> int:
    raw = get(name)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def resolve_path(value: str | Path) -> Path:
    """Make a configured path absolute, relative to the repo root.

    `.env.example` writes `CS_EVAL_DATA=data/conversations.yaml` because that is
    what a reader expects to see. Anchoring to ROOT rather than the process working
    directory means the command still works when run from elsewhere.
    """
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


def describe_source(name: str) -> str:
    """Where a value came from. Distinguishing these two matters: an exported
    variable silently outranks .env, and without this label the two are
    indistinguishable in the validate output."""
    if name in _FROM_FILE:
        return ".env"
    if os.environ.get(name):
        return "exported env"
    return "default"
