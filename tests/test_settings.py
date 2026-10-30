"""Tests for configuration loading and precedence. No AWS calls."""

from __future__ import annotations

import os

import pytest

from cs_eval import settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start each test with the settings-relevant variables unset."""
    for key in list(settings.DEFAULTS) + ["AWS_DEFAULT_REGION"]:
        monkeypatch.delenv(key, raising=False)


def test_falls_back_to_builtin_default():
    assert settings.get("AWS_REGION") == "us-east-1"
    assert settings.get("AWS_PROFILE") == ""  # no profile name is committed
    assert settings.describe_source("AWS_REGION") == "default"


def test_env_var_beats_default(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-central-1")
    assert settings.get("AWS_REGION") == "eu-central-1"
    assert settings.describe_source("AWS_REGION") == "exported env"


def test_env_file_is_read(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("AWS_REGION=ap-south-1\n", encoding="utf-8")
    assert settings.load_env(env_file) == env_file
    assert settings.get("AWS_REGION") == "ap-south-1"


def test_exported_variable_beats_env_file(tmp_path, monkeypatch):
    """load_dotenv(override=False): a one-off export must not need a file edit."""
    monkeypatch.setenv("AWS_REGION", "us-west-1")
    env_file = tmp_path / ".env"
    env_file.write_text("AWS_REGION=ap-south-1\n", encoding="utf-8")
    settings.load_env(env_file)
    assert settings.get("AWS_REGION") == "us-west-1"


def test_source_distinguishes_env_file_from_exported(tmp_path, monkeypatch):
    """An exported variable outranks .env silently; the label must expose that."""
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AWS_REGION=us-east-1\nAWS_PROFILE=example-profile\n", encoding="utf-8"
    )
    settings.load_env(env_file)
    assert settings.describe_source("AWS_REGION") == "exported env"
    assert settings.describe_source("AWS_PROFILE") == ".env"
    assert settings.describe_source("CS_EVAL_MODEL_ID") == "default"


def test_missing_env_file_is_not_an_error(tmp_path):
    """Replay mode must work with no .env at all."""
    assert settings.load_env(tmp_path / "nope.env") is None
    assert settings.get("AWS_REGION") == "us-east-1"


def test_region_is_mirrored_for_boto3(tmp_path):
    """boto3 reads AWS_REGION in some paths and AWS_DEFAULT_REGION in others."""
    env_file = tmp_path / ".env"
    env_file.write_text("AWS_REGION=ca-central-1\n", encoding="utf-8")
    settings.load_env(env_file)
    assert os.environ["AWS_REGION"] == "ca-central-1"
    assert os.environ["AWS_DEFAULT_REGION"] == "ca-central-1"


def test_committed_example_covers_every_setting():
    """.env.example must document every knob, or the docs silently drift."""
    text = (settings.ROOT / ".env.example").read_text(encoding="utf-8")
    for name in settings.DEFAULTS:
        assert f"{name}=" in text, f"{name} is missing from .env.example"


def test_example_values_match_builtin_defaults():
    """An unedited copy of .env.example must behave like no .env at all."""
    parsed = settings._parse_env_file(settings.ROOT / ".env.example")
    for name, default in settings.DEFAULTS.items():
        assert parsed[name] == default, f"{name} drifted from its built-in default"


def test_parser_handles_comments_quotes_and_export(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        "\n"
        "export AWS_PROFILE=prod\n"
        'AWS_REGION="us-east-2"   # trailing comment\n',
        encoding="utf-8",
    )
    parsed = settings._parse_env_file(env_file)
    assert parsed["AWS_PROFILE"] == "prod"
    assert parsed["AWS_REGION"] == "us-east-2"


def test_numeric_settings_are_typed(monkeypatch):
    monkeypatch.setenv("CS_EVAL_MAX_USD", "1.25")
    monkeypatch.setenv("CS_EVAL_LIMIT", "4")
    assert settings.get_float("CS_EVAL_MAX_USD") == 1.25
    assert settings.get_int("CS_EVAL_LIMIT") == 4


def test_bad_numeric_setting_is_rejected(monkeypatch):
    monkeypatch.setenv("CS_EVAL_MAX_USD", "lots")
    with pytest.raises(ValueError):
        settings.get_float("CS_EVAL_MAX_USD")


def test_relative_dataset_path_anchors_to_repo_root():
    """Commands must work when run from a directory other than the repo root."""
    resolved = settings.resolve_path("data/golden_set.yaml")
    assert resolved.is_absolute()
    assert resolved.exists()


def test_absolute_path_is_left_alone(tmp_path):
    assert settings.resolve_path(tmp_path / "x.yaml") == tmp_path / "x.yaml"


def test_unknown_setting_rejected():
    with pytest.raises(KeyError):
        settings.get("CS_EVAL_NOT_A_THING")
