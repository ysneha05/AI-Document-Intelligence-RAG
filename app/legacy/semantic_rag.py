import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from semantic_retriever import create_embedding
from ollama_llm import generate_answer


INDEX_PATH = Path("data/semantic_index.json")


def load_index():
    with open(INDEX_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def search(query, index, top_k=3, similarity_threshold=0.50):
    query_embedding = create_embedding(query)

    document_embeddings = np.array(
        [item["embedding"] for item in index]
    )

    scores = cosine_similarity(
        [query_embedding],
        document_embeddings
    )[0]

    ranked_indices = scores.argsort()[::-1]

    results = []

    for position in ranked_indices:
        score = float(scores[position])

        if score < similarity_threshold:
            break

        results.append({
            "chunk_id": index[position]["chunk_id"],
            "filename": index[position]["filename"],
            "chunk_number": index[position]["chunk_number"],
            "score": score,
            "text": index[position]["text"]
        })

        if len(results) >= top_k:
            break

    return results


def build_prompt(question, results):
    context = "\n\n".join([
        (
            f"[Source: {result['filename']} | "
            f"Chunk {result['chunk_number']}]\n"
            f"{result['text']}"
        )
        for result in results
    ])

    prompt = f"""
You are a document question-answering assistant.

Your task is to answer the user's question using ONLY the
DOCUMENT CONTEXT below.

STRICT RULES:

1. Use only facts explicitly stated in the DOCUMENT CONTEXT.
2. Do not use your own knowledge.
3. Do not guess or infer missing facts.
4. Do not change, calculate, combine, or reinterpret numerical values
   unless the document itself explicitly does so.
5. When the question asks for a number, copy the number exactly from
   the relevant document passage.
6. Before answering, identify the exact sentence or sentences in the
   context that answer the question.
7. If the context contains conflicting numbers, report the numbers
   exactly as stated and explain the difference using only the context.
8. If the answer cannot be found in the context, say:
   "I couldn't find the answer in the provided document."
9. Do not invent sources, citations, chunk numbers, or facts.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    return prompt


def main():
    print("Loading semantic index...")

    index = load_index()

    print(f"Loaded {len(index)} document chunks.")

    question = input("\nAsk a question about the documents: ")

    results = search(
        question,
        index,
        top_k=3
    )

    print("\nRetrieved semantic sources:")
    print("=" * 70)

    for result in results:
        print(
            f"{result['filename']} | "
            f"Chunk {result['chunk_number']} | "
            f"Similarity: {result['score']:.4f}"
        )

    if not results:
        print("\nNo relevant information found in the documents.")
        print("The question will not be sent to the language model.")

        print("\nSEMANTIC RAG ANSWER")
        print("=" * 70)
        print("I couldn't find the answer in the provided document.")

        print("\nSOURCES")
        print("=" * 70)
        print("No relevant sources found.")

        return

    prompt = build_prompt(question, results)

    print("\nGenerating answer with Qwen2.5...")

    answer = generate_answer(prompt)

    print("\nSEMANTIC RAG ANSWER")
    print("=" * 70)
    print(answer)

    print("\nSOURCES")
    print("=" * 70)

    for result in results:
        print(
            f"- {result['filename']} | "
            f"Chunk {result['chunk_number']} | "
            f"Similarity: {result['score']:.4f}"
        )


if __name__ == "__main__":
    main()