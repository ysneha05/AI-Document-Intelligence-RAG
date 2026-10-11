from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chunker import load_all_chunks


def load_chunks():
    return load_all_chunks()


def create_retriever(chunks):
    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix


def search(query, chunks, vectorizer, matrix, top_k=5):
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        matrix
    )[0]

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:
        results.append({
            "filename": chunks[index]["filename"],
            "chunk_number": chunks[index]["chunk_number"],
            "score": float(scores[index]),
            "text": chunks[index]["text"]
        })

    return results


if __name__ == "__main__":
    print("Loading documents...")

    chunks = load_chunks()

    print(f"Total chunks: {len(chunks)}")

    vectorizer, matrix = create_retriever(chunks)

    print("TF-IDF index created successfully!")

    question = input(
        "\nAsk a question about the documents: "
    )

    results = search(
        question,
        chunks,
        vectorizer,
        matrix,
        top_k=5
    )

    print("\nTop results:")
    print("=" * 70)

    for result in results:
        print(
            f"\nFile: {result['filename']}"
        )
        print(
            f"Chunk: {result['chunk_number']}"
        )
        print(
            f"Similarity: {result['score']:.4f}"
        )
        print("-" * 70)
        print(result["text"][:1000])