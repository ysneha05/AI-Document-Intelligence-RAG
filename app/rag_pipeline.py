import time
from urllib import response

from app.hybrid_retriever import (
    load_chunks,
    create_tfidf_index,
    load_semantic_index,
    hybrid_search
)

from langchain_ollama import ChatOllama
from app.config import TOP_K, RETRIEVAL_K, OLLAMA_BASE_URL

llm = ChatOllama(
    model="qwen2.5:3b",
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    num_predict=256,
)

def build_prompt(question, results):
    context = "\n\n".join(
        [
            (
                f"[Source: {result['filename']} | "
                f"Chunk {result['chunk_number']}]\n"
                f"{result['text']}"
            )
            for result in results
        ]
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the document context.

Rules:
1. Use only information explicitly stated in the context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Do not guess.
5. Answer the specific question that was asked.
6. If the question asks "why", explain the reasons or consequences
   stated in the context rather than focusing only on a statistic.
7. If the question asks for a number, give the exact number stated
   in the context.
8. If multiple retrieved passages are relevant, combine their
   information only when the relationship is explicitly supported
   by the context.
9. If the context does not contain enough information to answer,
   say:
   "I couldn't find enough information in the provided documents."
10. When the context contains multiple relevant effects, pollutants,
    causes, or categories, cover each relevant one rather than
    focusing only on the first passage.
11. Keep the answer concise and factual.
12. Do not include citations, source names, or chunk numbers in your answer.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    return prompt

def initialize_rag():
    print("Loading documents...")
    chunks = load_chunks()

    print(f"Total chunks: {len(chunks)}")

    if not chunks:
        print("No PDFs found. Starting with an empty document index.")
        return [], None, None, []

    print("\nCreating TF-IDF index...")
    vectorizer, matrix = create_tfidf_index(chunks)
    print("TF-IDF index created successfully!")

    print("\nLoading semantic index...")
    try:
        semantic_index = load_semantic_index()
    except FileNotFoundError:
        print("Semantic index not found. Build the index after adding PDFs.")
        semantic_index = []

    print(f"Semantic index loaded: {len(semantic_index)} chunks")

    return chunks, vectorizer, matrix, semantic_index

def ask_question(
    question,
    chunks,
    vectorizer,
    matrix,
    semantic_index
):
    print("\nSearching documents...")

    retrieval_timer = time.perf_counter()

    results = hybrid_search(
        question,
        chunks,
        vectorizer,
        matrix,
        semantic_index,
        top_k=TOP_K,
        retrieval_k=RETRIEVAL_K
    )

    if not results:
        return {
            "answer": (
                "I couldn't find the answer in the "
                "provided documents."
            ),
            "sources": []
        }

    prompt = build_prompt(
        question,
        results
    )

    print(
        "Retrieval time:",
        round(time.perf_counter() - retrieval_timer, 2),
        "seconds"
    )
    
    context_chars = sum(len(result["text"]) for result in results)

    print("\n--- Prompt Size Diagnostics ---")
    print("Retrieved chunks:", len(results))
    print("Retrieved text characters:", context_chars)
    print("Full prompt characters:", len(prompt))
    print("--------------------------------")

    print("\nGenerating answer with Qwen2.5...")
    print("(This may take a while on your CPU.)")

    response = llm.invoke(prompt)
    answer = response.content

    metadata = response.response_metadata

    print("\n--- LLM Generation Diagnostics ---")
    print("Stop reason:", metadata.get("done_reason", "not provided"))
    print("Prompt tokens:", metadata.get("prompt_eval_count", "not provided"))
    print("Generated tokens:", metadata.get("eval_count", "not provided"))
    print("Prompt evaluation (ns):", metadata.get("prompt_eval_duration", "not provided"))
    print("Generation time (ns):", metadata.get("eval_duration", "not provided"))
    print("----------------------------------")

    sources = []

    for result in results:
        sources.append({
            "filename": result["filename"],
            "chunk_number": result["chunk_number"],
            "page_numbers": result.get("page_numbers", []),
            "page_start": result.get("page_start"),
            "page_end": result.get("page_end"),
            "score": result["final_score"]
        })

    return {
        "answer": answer,
        "sources": sources
    }

if __name__ == "__main__":
    chunks, vectorizer, matrix, semantic_index = initialize_rag()

    print("\nRAG Assistant is ready.")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        result = ask_question(
            question,
            chunks,
            vectorizer,
            matrix,
            semantic_index
        )

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:")
        for source in result["sources"]:
            print(
                f"- {source['filename']} | "
                f"Chunk {source['chunk_number']} | "
                f"Score: {source['score']:.4f}"
            )