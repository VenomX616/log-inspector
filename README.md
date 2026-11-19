# Log Inspector

A lightweight Python application that analyses
application log files and counts log severity levels.

## Features

- Counts DEBUG, INFO, WARNING and ERROR messages.
- Supports uppercase and lowercase severity labels.
- Recognises plain and bracketed severity labels.
- Counts at most one severity per line.
- Ignores lines without a recognised leading label.
- Supports iterable inputs.
- Reads log files using UTF-8 encoding.
- Includes automated pytest tests.

## Project Structure

```text
log-inspector/
├── .gitignore
├── README.md
├── requirements-dev.txt
└── log-inspector/
    ├── log_inspector.py
    └── tests/
        └── test_log_inspector.py