import json
import time
import statistics

from app.rag_pipeline import initialize_rag, ask_question


QUESTIONS_FILE = "evaluation/questions.json"
RESULTS_FILE = "evaluation/results.json"

def calculate_keyword_coverage(item, answer):
    expected_keywords = item.get("expected_keywords", [])

    if not expected_keywords:
        return {
            "matched_keywords": [],
            "missing_keywords": [],
            "keyword_coverage_percent": None
        }

    answer_lower = answer.lower()

    matched_keywords = [
        keyword
        for keyword in expected_keywords
        if keyword.lower() in answer_lower
    ]

    missing_keywords = [
        keyword
        for keyword in expected_keywords
        if keyword.lower() not in answer_lower
    ]

    coverage = (
        len(matched_keywords) / len(expected_keywords)
    ) * 100

    return {
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "keyword_coverage_percent": round(coverage, 2)
    }

def evaluate_result(item, result):
    answer = result["answer"]
    sources = result["sources"]

    if not answer:
        return False

    answer_lower = answer.lower()

    # ---------------------------------------------------------
    # UNANSWERABLE QUESTIONS
    # ---------------------------------------------------------
    if not item["answerable"]:
        refusal_phrases = [
            "couldn't find",
            "could not find",
            "not enough information",
            "don't have enough information",
            "do not have enough information",
        ]

        refusal_detected = any(
            phrase in answer_lower
            for phrase in refusal_phrases
        )

        return refusal_detected and len(sources) == 0

    # ---------------------------------------------------------
    # ANSWERABLE QUESTIONS
    # ---------------------------------------------------------
    if not sources:
        return False

    refusal_phrases = [
        "couldn't find the answer",
        "could not find the answer",
        "not enough information",
        "don't have enough information",
        "do not have enough information",
    ]

    if any(
        phrase in answer_lower
        for phrase in refusal_phrases
    ):
        return False

    # ---------------------------------------------------------
    # KEYWORD-BASED FACT CHECK
    # ---------------------------------------------------------
    expected_keywords = item.get("expected_keywords", [])

    if expected_keywords:
        matched_keywords = [
            keyword
            for keyword in expected_keywords
            if keyword.lower() in answer_lower
        ]

        keyword_coverage = (
            len(matched_keywords) / len(expected_keywords)
        )

        # Require at least 75% of expected facts.
        return keyword_coverage >= 0.75
    # ---------------------------------------------------------
    # FALLBACK FOR QUESTIONS WITHOUT KEYWORDS
    # ---------------------------------------------------------
    expected_answer = item.get("expected_answer")

    if expected_answer:
        expected_lower = expected_answer.lower()

        return (
            expected_lower in answer_lower
            or any(
                word in answer_lower
                for word in expected_lower.split()
                if len(word) > 4
            )
        )

    return True


def calculate_metrics(results):
    total = len(results)

    passed = sum(
        1 for result in results
        if result["passed"]
    )

    failed = total - passed

    answerable_results = [
        result
        for result in results
        if result["answerable"]
    ]

    answerable_with_sources = sum(
        1 for result in answerable_results
        if len(result["sources"]) > 0
    )

    unanswerable_results = [
        result
        for result in results
        if not result["answerable"]
    ]

    correct_refusals = sum(
        1
        for result in unanswerable_results
        if result["passed"]
    )

    response_times = [
        result["response_time_seconds"]
        for result in results
        if result["response_time_seconds"] is not None
    ]

    source_counts = [
        len(result["sources"])
        for result in results
    ]

    metrics = {
        "total_questions": total,
        "passed": passed,
        "failed": failed,
        "functional_pass_rate_percent": round(
            (passed / total) * 100,
            2
        ) if total else 0,

        "answerable_questions": len(answerable_results),

        "answerable_questions_with_sources": answerable_with_sources,

        "retrieval_success_rate_percent": round(
            (answerable_with_sources / len(answerable_results)) * 100,
            2
        ) if answerable_results else 0,

        "unanswerable_questions": len(unanswerable_results),

        "correct_refusal_count": correct_refusals,

        "correct_refusal_rate_percent": round(
            (correct_refusals / len(unanswerable_results)) * 100,
            2
        ) if unanswerable_results else 0,

        "average_response_time_seconds": round(
            statistics.mean(response_times),
            2
        ) if response_times else 0,

        "median_response_time_seconds": round(
            statistics.median(response_times),
            2
        ) if response_times else 0,

        "max_response_time_seconds": round(
            max(response_times),
            2
        ) if response_times else 0,

        "average_sources_per_question": round(
            statistics.mean(source_counts),
            2
        ) if source_counts else 0,
    }

    return metrics


def main():
    print("Loading evaluation questions...")

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    print(f"Total evaluation questions: {len(questions)}")

    print("\nInitializing RAG pipeline...")

    chunks, vectorizer, matrix, semantic_index = initialize_rag()

    results = []

    for item in questions:
        question_id = item["id"]
        question = item["question"]

        print("\n" + "=" * 70)
        print(f"Question {question_id}/{len(questions)}")
        print(question)

        start_time = time.perf_counter()

        try:
            result = ask_question(
                question,
                chunks,
                vectorizer,
                matrix,
                semantic_index
            )

            elapsed_time = time.perf_counter() - start_time

            passed = evaluate_result(
                item,
                result
            )

            keyword_evaluation = calculate_keyword_coverage(
                item,
                result["answer"]
            )

            evaluation_result = {
                "id": question_id,
                "question": question,
                "expected_answer": item["expected_answer"],
                "expected_keywords": item.get("expected_keywords", []),
                "generated_answer": result["answer"],
                "answerable": item["answerable"],
                "sources": result["sources"],
                "matched_keywords": keyword_evaluation["matched_keywords"],
                "missing_keywords": keyword_evaluation["missing_keywords"],
                "keyword_coverage_percent": keyword_evaluation["keyword_coverage_percent"], 
                "response_time_seconds": round(
                    elapsed_time,
                    2
                ),
                "passed": passed,
                "error": None
            }

        except Exception as e:
            elapsed_time = time.perf_counter() - start_time

            evaluation_result = {
                "id": question_id,
                "question": question,
                "expected_answer": item["expected_answer"],
                "answerable": item["answerable"],
                "generated_answer": None,
                "sources": [],
                "response_time_seconds": round(
                    elapsed_time,
                    2
                ),
                "passed": False,
                "error": str(e)
            }

        results.append(evaluation_result)

        status = (
            "PASS"
            if evaluation_result["passed"]
            else "FAIL"
        )

        print(f"\nStatus: {status}")
        print(
            f"Answer: "
            f"{evaluation_result['generated_answer']}"
        )
        print(
            f"Sources: "
            f"{len(evaluation_result['sources'])}"
        )
        print(
            f"Response time: "
            f"{evaluation_result['response_time_seconds']} seconds"
        )

        if evaluation_result["error"]:
            print(
                f"Error: "
                f"{evaluation_result['error']}"
            )

    metrics = calculate_metrics(results)

    output = {
        "metrics": metrics,
        "results": results
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("Evaluation Metrics")
    print("=" * 70)

    for key, value in metrics.items():
        print(
            f"{key}: {value}"
        )

    print("=" * 70)
    print("Evaluation completed.")
    print(
        f"Results saved to: "
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()