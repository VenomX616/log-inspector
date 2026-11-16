import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from log_inspector import summarize_log_file, summarize_log_lines


def test_empty_input_returns_every_level_with_zero_count() -> None:
    assert summarize_log_lines([]) == {
        "DEBUG": 0,
        "INFO": 0,
        "WARNING": 0,
        "ERROR": 0,
    }


def test_counts_all_severity_levels() -> None:
    lines = [
        "DEBUG startup",
        "INFO request accepted",
        "WARNING slow request",
        "ERROR request failed",
        "INFO another request accepted",
    ]

    assert summarize_log_lines(lines) == {
        "DEBUG": 1,
        "INFO": 2,
        "WARNING": 1,
        "ERROR": 1,
    }


@pytest.mark.parametrize(
    ("line", "expected_level"),
    [
        ("debug details", "DEBUG"),
        ("[info] ready", "INFO"),
        ("warning: low disk space", "WARNING"),
        ("ERROR request ended", "ERROR"),
        ("iNfO mixed case", "INFO"),
    ],
)
def test_matches_labels_without_regard_to_case(
    line: str,
    expected_level: str,
) -> None:
    counts = summarize_log_lines([line])

    assert counts[expected_level] == 1
    assert sum(counts.values()) == 1


def test_ignores_malformed_unknown_and_embedded_labels() -> None:
    lines = [
        "",
        "not a formatted log line",
        "TRACE details",
        "INFORMATION is not a severity label",
        "prefixERRORsuffix",
    ]

    assert summarize_log_lines(lines) == {
        "DEBUG": 0,
        "INFO": 0,
        "WARNING": 0,
        "ERROR": 0,
    }


def test_malformed_line_with_severity_is_ignored() -> None:
    lines = [
        "[ERROR] Database failed",
        "not a valid log record ERROR ???",
    ]

    result = summarize_log_lines(lines)

    assert result["ERROR"] == 1


def test_counts_only_the_first_recognized_label_on_each_line() -> None:
    lines = [
        "INFO then ERROR",
        "[WARNING] later DEBUG",
        "ERROR ERROR",
    ]

    assert summarize_log_lines(lines) == {
        "DEBUG": 0,
        "INFO": 1,
        "WARNING": 1,
        "ERROR": 1,
    }


def test_accepts_one_shot_iterable() -> None:
    lines = (line for line in ["info ready", "ERROR failed", "unknown"])

    assert summarize_log_lines(lines) == {
        "DEBUG": 0,
        "INFO": 1,
        "WARNING": 0,
        "ERROR": 1,
    }


def test_summarize_log_file_reads_utf8_text(tmp_path: Path) -> None:
    log_file = tmp_path / "application.txt"
    log_file.write_text(
        "[DEBUG] starting\ninfo ready\n[ERROR] request failed\n"
        "malformed ERROR ???\n",
        encoding="utf-8",
    )

    assert summarize_log_file(log_file) == {
        "DEBUG": 1,
        "INFO": 1,
        "WARNING": 0,
        "ERROR": 1,
    }


def test_summarize_log_file_accepts_string_path(tmp_path: Path) -> None:
    log_file = tmp_path / "application.txt"
    log_file.write_text("WARNING: low disk space\n", encoding="utf-8")

    assert summarize_log_file(str(log_file)) == {
        "DEBUG": 0,
        "INFO": 0,
        "WARNING": 1,
        "ERROR": 0,
    }


def test_summarize_empty_log_file_returns_zero_counts(tmp_path: Path) -> None:
    log_file = tmp_path / "empty.txt"
    log_file.write_text("", encoding="utf-8")

    assert summarize_log_file(log_file) == {
        "DEBUG": 0,
        "INFO": 0,
        "WARNING": 0,
        "ERROR": 0,
    }


def test_summarize_log_file_propagates_missing_file_error(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        summarize_log_file(missing_file)
