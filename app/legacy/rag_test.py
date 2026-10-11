from rag_prompt import load_chunks, create_retriever, search
from ollama_llm import generate_answer


def build_rag_prompt(question, results):
    context = "\n\n".join(
        [
            f"[Source Chunk {result['chunk_id']}]\n{result['text']}"
            for result in results
        ]
    )

    prompt = f"""
You are an AI document assistant.

Answer the user's question using ONLY the document context
provided below.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not in the context, say:
   "I couldn't find the answer in the provided document."
4. At the end, list the source chunk numbers you used.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    return prompt


if __name__ == "__main__":
    print("Loading document...")

    chunks = load_chunks()
    vectorizer, matrix = create_retriever(chunks)

    print(f"Total chunks: {len(chunks)}")

    question = input("\nAsk a question about the document: ")

    results = search(
        question,
        chunks,
        vectorizer,
        matrix,
        top_k=3
    )

    print("\nRetrieved sources:")
    print("=" * 70)

    for result in results:
        print(
            f"Chunk {result['chunk_id']} "
            f"(similarity: {result['score']:.4f})"
        )

    prompt = build_rag_prompt(question, results)

    print("\nGenerating answer with Qwen2.5...")
    answer = generate_answer(prompt)

    print("\nRAG ANSWER")
    print("=" * 70)
    print(answer)