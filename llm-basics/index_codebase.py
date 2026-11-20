
from pathlib import Path

import chromadb
import ollama


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DB_PATH = Path(__file__).resolve().parent / "chroma_db"

COLLECTION_NAME = "log_inspector_knowledge"

EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_LINES = 45
CHUNK_OVERLAP = 8

SOURCE_FILES = [
    "log-inspector/log_inspector.py",
    "log-inspector/tests/test_log_inspector.py",
]


# --------------------------------------------------
# 2. Document chunking
# --------------------------------------------------

def split_into_chunks(relative_path: str, text: str) -> list[dict]:
    """Split a source file into overlapping line chunks."""

    lines = text.splitlines()

    chunks = []

    step = CHUNK_LINES - CHUNK_OVERLAP

    for start in range(0, len(lines), step):

        end = min(start + CHUNK_LINES, len(lines))

        content = "\n".join(lines[start:end]).strip()

        if not content:
            continue

        start_line = start + 1
        end_line = end

        document = (
            f"Source file: {relative_path}\n"
            f"Lines: {start_line}-{end_line}\n\n"
            f"{content}"
        )

        chunks.append({
            "id": (
                f"{relative_path.replace('/', '_')}"
                f"_{start_line}_{end_line}"
            ),
            "document": document,
            "source": relative_path,
            "start_line": start_line,
            "end_line": end_line,
        })

        if end >= len(lines):
            break

    return chunks


# --------------------------------------------------
# 3. Read source files
# --------------------------------------------------

def load_project_files() -> list[dict]:

    all_chunks = []

    for relative_path in SOURCE_FILES:

        file_path = ROOT / relative_path

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Source file not found: {file_path}"
            )

        text = file_path.read_text(encoding="utf-8-sig")

        chunks = split_into_chunks(relative_path, text)

        all_chunks.extend(chunks)

        print(
            f"Read {relative_path}: {len(chunks)} chunks"
        )

    return all_chunks


# --------------------------------------------------
# 4. Main indexing process
# --------------------------------------------------

def index_codebase() -> None:

    print("\nStarting codebase indexing...")

    chunks = load_project_files()

    if not chunks:
        print("No documents found.")
        return

    # Connect to local persistent ChromaDB.
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Remove old records before rebuilding the index.
    existing = collection.get()

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    print("\nGenerating embeddings...")

    # IMPORTANT:
    # Nomic requires a task prefix for retrieval.
    documents_for_embedding = [
        f"search_document: {chunk['document']}"
        for chunk in chunks
    ]

    embedding_response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=documents_for_embedding,
    )

    embeddings = embedding_response.embeddings

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Embedding count does not match chunk count."
        )

    print("Saving information to ChromaDB...")

    collection.add(
        ids=[chunk["id"] for chunk in chunks],

        documents=[
            chunk["document"] for chunk in chunks
        ],

        embeddings=[
            list(vector) for vector in embeddings
        ],

        metadatas=[
            {
                "source": chunk["source"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
            }
            for chunk in chunks
        ],
    )

    print("\nIndexing completed!")

    print(f"Total chunks stored: {collection.count()}")

    print(f"Database location: {DB_PATH}")


if __name__ == "__main__":
    index_codebase()
