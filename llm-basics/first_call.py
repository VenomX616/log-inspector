
from pathlib import Path

from ollama import chat, ResponseError

MODEL = "llama3.2:latest"

# The root folder of ai-engineering-lab.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Only provide the files relevant to this question.
PROJECT_FILES = [
    "log-inspector/log_inspector.py",
    "log-inspector/tests/test_log_inspector.py",
]


def load_project_context() -> str:
    """Read selected project files and prepare LLM context."""

    sections = []

    for relative_path in PROJECT_FILES:
        file_path = PROJECT_ROOT / relative_path

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Project file not found: {file_path}"
            )

        source = file_path.read_text(encoding="utf-8")

        sections.append(
            f"FILE: {relative_path}\n"
            f"```python\n{source}\n```"
        )

    return "\n\n".join(sections)


def ask_tutor(question: str) -> None:
    """Answer questions using actual project source code."""

    try:
        context = load_project_context()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful Python codebase analyst. "
                    "Use only the supplied source files as evidence "
                    "for claims about this project. "
                    "Do not invent features, files, dependencies "
                    "or functionality. "
                    "If something is not shown in the source, "
                    "explicitly say that it cannot be confirmed. "
                    "Refer to actual file paths and function names."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PROJECT SOURCE CODE:\n\n{context}\n\n"
                    f"MY QUESTION:\n{question}"
                ),
            },
        ]

        print("\nReading your project and thinking...\n")

        response = chat(
            model=MODEL,
            messages=messages,
        )

        print("\n--- GROUNDED AI RESPONSE ---\n")
        print(response.message.content)

        print("\n--- TOKEN USAGE ---")
        print(f"Input tokens: {response.prompt_eval_count}")
        print(f"Output tokens: {response.eval_count}")

    except FileNotFoundError as exc:
        print(f"File error: {exc}")

    except ResponseError as exc:
        print(f"Ollama error: {exc}")

    except Exception as exc:
        print(f"Unexpected error: {exc}")


def main() -> None:
    print("\nAI Python Tutor - Codebase Aware Version")

    question = input(
        "\nAsk a question about your project: "
    ).strip()

    if question:
        ask_tutor(question)
    else:
        print("Please enter a question.")


if __name__ == "__main__":
    main()
