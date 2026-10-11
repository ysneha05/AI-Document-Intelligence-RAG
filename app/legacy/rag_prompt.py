from tfidf_retriever import load_chunks, create_retriever, search


def build_prompt(question, results):
    context = "\n\n".join(
        [
            f"[Source {result['chunk_id']}]\n{result['text']}"
            for result in results
        ]
    )

    prompt = f"""
You are an AI document assistant.

Answer the user's question using ONLY the information provided
in the document context below.

If the answer cannot be found in the context, say:
"I couldn't find the answer in the provided document."

Do not invent information.

Always mention the source chunk(s) used.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    return prompt


if __name__ == "__main__":
    chunks = load_chunks()
    vectorizer, matrix = create_retriever(chunks)

    question = input("\nAsk a question about the document: ")

    results = search(
        question,
        chunks,
        vectorizer,
        matrix,
        top_k=3
    )

    prompt = build_prompt(question, results)

    print("\nGenerated RAG Prompt:")
    print("=" * 70)
    print(prompt)