import json
from pathlib import Path

from app.chunker import load_all_chunks
from app.semantic_retriever import create_embedding
from app.config import SEMANTIC_INDEX_PATH

INDEX_PATH = SEMANTIC_INDEX_PATH

def build_index():
    print("Loading documents...")
    chunks = load_all_chunks()

    print(f"Total chunks: {len(chunks)}")
    print("\nCreating semantic embeddings...")

    index = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"Embedding chunk {i}/{len(chunks)}...")

        embedding = create_embedding(chunk["text"])

        index.append({
            "chunk_id": i,
            "filename": chunk["filename"],
            "chunk_number": chunk["chunk_number"],
            "page_numbers": chunk.get("page_numbers", []),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "text": chunk["text"],
            "embedding": embedding
        })

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(INDEX_PATH, "w", encoding="utf-8") as file:
        json.dump(index, file)

    print("\nSemantic index saved successfully!")
    print(f"Location: {INDEX_PATH}")
    print(f"Chunks indexed: {len(index)}")

def load_semantic_index():
    if not INDEX_PATH.exists():
        print("Semantic index not found. Returning an empty index.")
        return []

    with open(INDEX_PATH, "r", encoding="utf-8") as file:
        return json.load(file)
    
if __name__ == "__main__":
    build_index()