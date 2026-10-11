import json
from pathlib import Path
from unittest import result

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.chunker import load_all_chunks
from app.semantic_retriever import create_embedding


from app.config import (
    SEMANTIC_INDEX_PATH,
    TOP_K,
    RETRIEVAL_K,
    RELEVANCE_THRESHOLD
)

INDEX_PATH = SEMANTIC_INDEX_PATH


def load_chunks():
    return load_all_chunks()


def create_tfidf_index(chunks):
    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix


def load_semantic_index():
    with open(INDEX_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def semantic_search(query, semantic_index, top_k=10):
    query_embedding = create_embedding(query)

    document_embeddings = np.array(
        [item["embedding"] for item in semantic_index]
    )

    scores = cosine_similarity(
        [query_embedding],
        document_embeddings
    )[0]

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []

    for rank, index in enumerate(ranked_indices, start=1):
        item = semantic_index[index]

        results.append({
            "chunk_id": item["chunk_id"],
            "filename": item["filename"],
            "chunk_number": item["chunk_number"],
            "page_numbers": item.get("page_numbers", []),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "text": item["text"],
            "semantic_score": float(scores[index]),
            "semantic_rank": rank
        })

    return results


def tfidf_search(query, chunks, vectorizer, matrix, top_k=10):
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        matrix
    )[0]

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []

    for rank, index in enumerate(ranked_indices, start=1):
        chunk = chunks[index]

        results.append({
            "chunk_id": index + 1,
            "filename": chunk["filename"],
            "chunk_number": chunk["chunk_number"],
            "page_numbers": chunk.get("page_numbers", []),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "text": chunk["text"],
            "tfidf_score": float(scores[index]),
            "tfidf_rank": rank
        })

    return results

def section_heading_boost(query, text):
    """
    Boost chunks where the requested section heading is followed
    by actual section content.
    """

    query_lower = query.lower()
    text_lower = text.lower()

    section_patterns = {
        "agriculture": "effects on agriculture",
        "environment": "effects on environment",
        "economy": "effects on economy",
        "planet": "long-term effects on the planet",
        "sustainable development": "effects on sustainable development goals",
    }

    for keyword, heading in section_patterns.items():

        if keyword not in query_lower:
            continue

        position = text_lower.find(heading)

        if position == -1:
            continue

        # Text immediately after the heading.
        after_heading = text_lower[
            position + len(heading):
            position + len(heading) + 300
        ]

        # A real section normally has substantive prose after
        # the heading. A TOC normally has another numbered
        # heading shortly afterward.
        if len(after_heading.split()) >= 20:
            return 0.025

    return 0.0

def find_section_chunks(query, chunks):
    """
    Find chunks containing the requested section heading.
    Prefer the actual section content over table-of-contents chunks.
    """

    query_lower = query.lower()

    section_patterns = {
        "agriculture": "effects on agriculture",
        "environment": "effects on envir",
        "economy": "effects on economy",
        "planet": "long-term effects on the planet",
        "sustainable development": "effects on sustainable development goals",
    }

    matched_heading = None

    for keyword, heading in section_patterns.items():
        if keyword in query_lower:
            matched_heading = heading
            break

    if matched_heading is None:
        return []

    results = []

    for chunk in chunks:
        text_lower = chunk["text"].lower()

        if matched_heading not in text_lower:
            continue

        # Count nearby section headings.
        section_markers = [
            "effects on agriculture",
            "effects on environment",
            "effects on economy",
            "long-term effects on the planet",
            "effects on sustainable development goals",
        ]

        nearby_sections = sum(
            marker in text_lower
            for marker in section_markers
        )

        # A table of contents usually contains several section
        # headings together. Ignore those chunks.
        if nearby_sections >= 2:
            continue

        results.append(chunk)

    return results


def hybrid_search(
    query,
    chunks,
    vectorizer,
    matrix,
    semantic_index,
    top_k=5,
    retrieval_k=10
):
    if not chunks:
        return []

    tfidf_results = tfidf_search(
        query,
        chunks,
        vectorizer,
        matrix,
        top_k=retrieval_k
    )

    semantic_results = semantic_search(
        query,
        semantic_index,
        top_k=retrieval_k
    )

    combined = {}

        # Targeted section discovery
    section_chunks = find_section_chunks(query, chunks)

    for chunk in section_chunks:
        key = (
            chunk["filename"],
            chunk["chunk_number"]
        )

        combined[key] = {
            "filename": chunk["filename"],
            "chunk_number": chunk["chunk_number"],
            "text": chunk["text"],
            "hybrid_score": 0.10
        }

    # Reciprocal Rank Fusion
    for result in tfidf_results:
        key = (
            result["filename"],
            result["chunk_number"]
        )

        combined.setdefault(
            key,
            {
                "filename": result["filename"],
                "chunk_number": result["chunk_number"],
                "text": result["text"],
                "hybrid_score": 0.0
            }
        )

        combined[key]["hybrid_score"] += (
            1 / (60 + result["tfidf_rank"])
        )

    for result in semantic_results:
        key = (
            result["filename"],
            result["chunk_number"]
        )

        combined.setdefault(
            key,
            {
                "filename": result["filename"],
                "chunk_number": result["chunk_number"],
                "text": result["text"],
                "hybrid_score": 0.0
            }
        )

        combined[key]["hybrid_score"] += (
            1 / (60 + result["semantic_rank"])
        )

    # Keyword and section boosts
    query_words = set(
        word.lower().strip(".,?!")
        for word in query.split()
        if len(word) >= 4
    )

    for result in combined.values():

        text_words = set(
            word.lower().strip(".,?!")
            for word in result["text"].split()
        )

        matched_words = query_words.intersection(text_words)

        result["keyword_matches"] = len(matched_words)

        heading_boost = section_heading_boost(
            query,
            result["text"]
        )

        result["heading_boost"] = heading_boost

        # Table-of-contents penalty
        toc_penalty = 0.0
        text_lower = result["text"].lower()

        toc_markers = [
            "effects on agriculture",
            "effects on environment",
            "effects on economy",
            "long-term effects on the planet",
            "effects on sustainable development goals",
        ]

        toc_marker_count = sum(
            marker in text_lower
            for marker in toc_markers
        )

        # Penalize only chunks that are primarily TOC-like.
        # Some chunks contain TOC text followed by real document content.
        word_count = len(result["text"].split())

        if toc_marker_count >= 3 and word_count < 80:
            toc_penalty = 0.030
        elif toc_marker_count >= 2 and word_count < 50:
            toc_penalty = 0.015

        result["toc_penalty"] = toc_penalty

        result["final_score"] = (
            result["hybrid_score"]
            + (0.001 * len(matched_words))
            + heading_boost
            - toc_penalty
        )

    # Rank original retrieval results
    ranked_results = sorted(
        combined.values(),
        key=lambda x: x["final_score"],
        reverse=True
    )

    # Relevance gate
    relevant_results = [
        result
        for result in ranked_results
        if result["final_score"] >= RELEVANCE_THRESHOLD
    ]

    # Select the strongest retrieval results first.
    seed_results = relevant_results[:top_k]

    # Add neighboring chunks.
    # Neighbors are included because a document section can span
    # multiple chunks, but they retain their own identity.
    expanded_results = []
    seen = set()

    for result in seed_results:

        filename = result["filename"]
        chunk_number = result["chunk_number"]

        for neighbor_number in [
            chunk_number - 1,
            chunk_number,
            chunk_number + 1
        ]:

            if neighbor_number < 1:
                continue

            key = (filename, neighbor_number)

            if key in seen:
                continue

            for chunk in chunks:

                if (
                    chunk["filename"] == filename
                    and chunk["chunk_number"] == neighbor_number
                ):

                    expanded_results.append({
                        "filename": chunk["filename"],
                        "chunk_number": chunk["chunk_number"],
                        "page_numbers": chunk.get("page_numbers", []),
                        "page_start": chunk.get("page_start"),
                        "page_end": chunk.get("page_end"),
                        "text": chunk["text"],
                        "hybrid_score": (
                            result["hybrid_score"]
                            if neighbor_number == chunk_number
                            else result["hybrid_score"] * 0.8
                        ),
                        "keyword_matches": (
                            result["keyword_matches"]
                            if neighbor_number == chunk_number
                            else 0
                        ),
                        "final_score": (
                            result["final_score"]
                            if neighbor_number == chunk_number
                            else result["final_score"] * 0.8
                        ),
                    })

                    seen.add(key)
                    break

    return expanded_results

if __name__ == "__main__":

    print("Loading documents...")
    chunks = load_chunks()

    print(f"Total chunks: {len(chunks)}")

    print("\nCreating TF-IDF index...")
    vectorizer, matrix = create_tfidf_index(chunks)

    print("TF-IDF index created successfully!")

    print("\nLoading semantic index...")
    semantic_index = load_semantic_index()

    print(
        f"Semantic index loaded: "
        f"{len(semantic_index)} chunks"
    )

    question = input(
        "\nAsk a question about the documents: "
    )

    results = hybrid_search(
        question,
        chunks,
        vectorizer,
        matrix,
        semantic_index,
        top_k=5,
        retrieval_k=10
    )

    print("\nHYBRID RETRIEVAL RESULTS")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):

        print(
            f"\nRank: {rank}\n"
            f"File: {result['filename']}\n"
            f"Chunk: {result['chunk_number']}\n"
            f"Hybrid Score: {result['hybrid_score']:.6f}\n"
            f"Keyword Matches: {result['keyword_matches']}\n"
            f"Final Score: {result['final_score']:.6f}"
        )

        print("-" * 70)
        print(result["text"][:1000])