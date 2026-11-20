
import json
import sys
from pathlib import Path
from typing import Any

from ollama import chat, ResponseError


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL = "llama3.2:latest"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

LOG_PROJECT_DIR = PROJECT_ROOT / "log-inspector"
SAMPLE_LOG_PATH = BASE_DIR / "sample.log"

sys.path.insert(0, str(LOG_PROJECT_DIR))

from log_inspector import summarize_log_file


# --------------------------------------------------
# 2. Calculator
# --------------------------------------------------

def calculate_average(numbers: list[int | float]) -> float:
    """Calculate the average of numeric values."""

    if not numbers:
        raise ValueError("Cannot calculate an empty list.")

    if any(
        isinstance(n, bool)
        or not isinstance(n, (int, float))
        for n in numbers
    ):
        raise ValueError("All values must be numbers.")

    return sum(numbers) / len(numbers)


# --------------------------------------------------
# 3. Log analyser
# --------------------------------------------------

def analyze_sample_log() -> dict[str, int]:
    """Analyse the predefined sample log."""

    if not SAMPLE_LOG_PATH.is_file():
        raise FileNotFoundError("Sample log file not found.")

    return summarize_log_file(str(SAMPLE_LOG_PATH))


# --------------------------------------------------
# 4. Composite tool
# --------------------------------------------------

def analyze_and_average_sample_log() -> dict[str, Any]:
    """
    Analyse log severity counts and calculate
    the average of INFO and ERROR counts.
    """

    # First operation: analyse the file.
    counts = analyze_sample_log()

    # Second operation: use actual Python results.
    selected_counts = [
        counts["INFO"],
        counts["ERROR"],
    ]

    average = calculate_average(selected_counts)

    return {
        "severity_counts": counts,
        "average_of_info_and_error": average,
        "calculation": (
            f"({counts['INFO']} + {counts['ERROR']}) / 2 "
            f"= {average}"
        ),
    }


# --------------------------------------------------
# 5. Register the composite tool
# --------------------------------------------------

TOOLS = [analyze_and_average_sample_log]

AVAILABLE_TOOLS = {
    "analyze_and_average_sample_log":
        analyze_and_average_sample_log
}


# --------------------------------------------------
# 6. Safely execute tools
# --------------------------------------------------


def execute_tool(tool_call) -> dict[str, Any]:
    """Execute an explicitly registered, approved tool."""

    name = tool_call.function.name
    arguments = dict(tool_call.function.arguments or {})

    print(f"Received arguments: {arguments!r}")

    function = AVAILABLE_TOOLS.get(name)

    if function is None:
        return {
            "ok": False,
            "error": f"Unknown tool: {name}"
        }

    # This composite tool has no parameters.
    # Extra model-generated arguments are irrelevant
    # to this fixed, predefined operation.
    if name == "analyze_and_average_sample_log":

        if arguments:
            print(
                "Note: Ignoring unexpected arguments "
                "for this fixed operation."
            )

        try:
            result = function()

            return {
                "ok": True,
                "result": result
            }

        except (ValueError, FileNotFoundError) as exc:
            return {
                "ok": False,
                "error": str(exc)
            }

    return {
        "ok": False,
        "error": "Tool execution is not configured."
    }



# --------------------------------------------------
# 7. AI interaction
# --------------------------------------------------

def ask_ai(question: str) -> None:

    messages = [
        {
            "role": "system",
            "content": (
                "You are a reliable Python AI assistant. "
                "You have access to one tool: "
                "analyze_and_average_sample_log. "

                "This tool accepts NO arguments. "
                "Always invoke it with an empty arguments object. "

                "It analyses the predefined sample log and calculates "
                "the average of INFO and ERROR counts internally. "

                "Never invent log data or tool results. "
                "Use only the actual returned result."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    try:

        response = chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS
        )

        messages.append(response.message)

        tool_calls = response.message.tool_calls or []

        if not tool_calls:
            print("\nNo tool was called.")
            print(response.message.content)
            return

        # Execute the registered tool.
        tool_results = []

        for tool_call in tool_calls:

            print(
                f"\nCalling tool: {tool_call.function.name}"
            )

            result = execute_tool(tool_call)

            print(f"Tool result: {json.dumps(result)}")

            # Never ask the LLM to invent a result after failure.
            if not result["ok"]:
                print("\nWORKFLOW STOPPED: Tool execution failed.")
                print(result["error"])
                return

            tool_results.append(result["result"])

            messages.append({
                "role": "tool",
                "tool_name": tool_call.function.name,
                "content": json.dumps(result)
            })

        # Ask the LLM to explain successful results.
        final_response = chat(
            model=MODEL,
            messages=messages
        )

        print("\n--- AI EXPLANATION ---\n")
        print(final_response.message.content)

        # Show authoritative application results separately.
        print("\n--- VERIFIED PYTHON RESULTS ---\n")

        for result in tool_results:

            print("Severity counts:")

            for level, count in result["severity_counts"].items():
                print(f"{level}: {count}")

            print(
                "\nAverage of INFO and ERROR: "
                f"{result['average_of_info_and_error']}"
            )

            print(
                "Calculation: "
                f"{result['calculation']}"
            )

    except ResponseError as exc:
        print(f"Ollama error: {exc}")

    except Exception as exc:
        print(f"Unexpected error: {exc}")


# --------------------------------------------------
# 8. Entry point
# --------------------------------------------------

def main() -> None:

    print("\nWelcome to your Improved Python AI Agent!")

    question = input("\nAsk something: ").strip()

    if question:
        ask_ai(question)
    else:
        print("Please enter a question.")


if __name__ == "__main__":
    main()
