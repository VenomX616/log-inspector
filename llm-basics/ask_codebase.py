
from pathlib import Path

import chromadb
import ollama


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "log_inspector_knowledge"

CHAT_MODEL = "llama3.2:latest"

EMBEDDING_MODEL = "nomic-embed-text"

TOP_K = 6


# --------------------------------------------------
# 2. Connect to ChromaDB
# --------------------------------------------------

def get_collection():

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    if collection.count() == 0:
        raise RuntimeError(
            "Knowledge base is empty. Run index_codebase.py first."
        )

    return collection


# --------------------------------------------------
# 3. Retrieve relevant documents
# --------------------------------------------------

def retrieve_documents(question: str) -> dict:

    # Convert the user's question into an embedding.
    embedding_response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=f"search_query: {question}",
    )

    query_vector = list(embedding_response.embeddings[0])

    collection = get_collection()

    number_of_results = min(
        TOP_K,
        collection.count()
    )

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=number_of_results,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    return results


# --------------------------------------------------
# 4. Build the context
# --------------------------------------------------

def build_context(results: dict) -> str:

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    sections = []

    for index, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1,
    ):

        source = metadata["source"]
        start = metadata["start_line"]
        end = metadata["end_line"]

        sections.append(
            f"SOURCE {index}\n"
            f"File: {source}\n"
            f"Lines: {start}-{end}\n\n"
            f"{document}"
        )

    return "\n\n".join(sections)


# --------------------------------------------------
# 5. Generate a grounded response
# --------------------------------------------------

def ask_question(question: str) -> None:

    print("\nSearching your codebase...")

    results = retrieve_documents(question)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        print("No relevant documents found.")
        return

    context = build_context(results)

    print("\nRetrieved sources:")

    for metadata, distance in zip(metadatas, distances):

        print(
            f"- {metadata['source']} "
            f"(lines {metadata['start_line']}-"
            f"{metadata['end_line']}, "
            f"cosine distance: {distance:.4f})"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful Python codebase analyst.\n\n"

                "Use only the retrieved source code as evidence.\n\n"

                "Important rules:\n"
                "1. Distinguish implementation files from test files.\n"
                "2. Never infer implementation details from test names alone.\n"
                "3. Never invent or reconstruct code that is not provided.\n"
                "4. Do not claim that a function calls another function "
                "unless the implementation demonstrates it.\n"
                "5. Do not confuse test setup operations with application "
                "operations.\n"
                "6. Cite the actual source file and line range.\n"
                "7. If implementation evidence is missing, say so clearly.\n"
                "8. Separate verified facts from reasonable inferences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"RETRIEVED PROJECT CONTEXT:\n\n"
                f"{context}\n\n"
                f"QUESTION:\n{question}"
            ),
        },
    ]

    print("\nGenerating answer with Llama 3.2...")

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
    )

    print("\n" + "=" * 55)
    print("RAG ASSISTANT RESPONSE")
    print("=" * 55)

    print(response.message.content)

    print("\n" + "=" * 55)
    print("MODEL TOKEN USAGE")
    print("=" * 55)

    print(f"Input tokens: {response.prompt_eval_count}")
    print(f"Output tokens: {response.eval_count}")


# --------------------------------------------------
# 6. Application entry point
# --------------------------------------------------

def main():

    print("\nWelcome to your Local RAG Codebase Assistant!")

    question = input(
        "\nAsk something about your project: "
    ).strip()

    if not question:
        print("Please enter a question.")
        return

    try:
        ask_question(question)

    except Exception as exc:
        print(f"\nApplication error: {exc}")


if __name__ == "__main__":
    main()
