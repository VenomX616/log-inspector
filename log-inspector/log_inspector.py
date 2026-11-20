import re
from collections.abc import Iterable
from pathlib import Path


_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
_SEVERITY_PATTERN = re.compile(
    r"^\s*(?:\[(DEBUG|INFO|WARNING|ERROR)\]|(DEBUG|INFO|WARNING|ERROR))(?=$|[\s:])",
    re.IGNORECASE,
)


def summarize_log_lines(lines: Iterable[str]) -> dict[str, int]:
    """Count leading severity labels, counting at most one label per line.

    A label must be the first token, either plain or enclosed in brackets.
    Matching is case-insensitive. Lines without a leading recognized label are
    ignored.
    """
    counts = dict.fromkeys(_LEVELS, 0)
    for line in lines:
        match = _SEVERITY_PATTERN.search(line)
        if match is not None:
            level = next(group for group in match.groups() if group is not None)
            level = level.upper()
            counts[level] += 1
    return counts


def summarize_log_file(path: str | Path) -> dict[str, int]:
    """Read a UTF-8 text log file and summarize its severity counts.

    File access and decoding errors are propagated to the caller.
    """
    with Path(path).open(encoding="utf-8-sig") as log_file:
        return summarize_log_lines(log_file)
