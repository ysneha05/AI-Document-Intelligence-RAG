import requests
from sklearn.metrics.pairwise import cosine_similarity

from app.chunker import load_all_chunks
from app.config import OLLAMA_EMBEDDING_URL, EMBEDDING_MODEL


def create_embedding(text):
    response = requests.post(
        OLLAMA_EMBEDDING_URL,
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=120
    )
    response.raise_for_status()

    return response.json()["embedding"]


def load_chunks():
    return load_all_chunks()


def create_embedding_index(chunks):
    print("Creating embeddings for document chunks...")

    embeddings = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"Embedding chunk {i}/{len(chunks)}...")
        embedding = create_embedding(chunk["text"])
        embeddings.append(embedding)

    return embeddings


def search(query, chunks, embeddings, top_k=3, similarity_threshold=0.50):
    query_embedding = create_embedding(query)

    scores = cosine_similarity(
        [query_embedding],
        embeddings
    )[0]

    ranked_indices = scores.argsort()[::-1]

    results = []

    for index in ranked_indices:
        score = float(scores[index])

        if score < similarity_threshold:
            break

        results.append({
            "chunk_id": int(index + 1),
            "filename": chunks[index]["filename"],
            "chunk_number": chunks[index]["chunk_number"],
            "score": score,
            "text": chunks[index]["text"]
        })

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    print("Loading documents...")

    chunks = load_chunks()

    print(f"Total chunks: {len(chunks)}")

    embeddings = create_embedding_index(chunks)

    print("\nSemantic index created successfully!")

    query = input("\nAsk a question about the documents: ")

    results = search(
        query,
        chunks,
        embeddings,
        top_k=3
    )

    print("\nTop semantic results:")
    print("=" * 70)

    for result in results:
        print(f"\nFile: {result['filename']}")
        print(f"Chunk: {result['chunk_number']}")
        print(f"Similarity: {result['score']:.4f}")
        print("-" * 70)
        print(result["text"][:700])