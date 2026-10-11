
from fastapi.testclient import TestClient
from unittest.mock import patch

# Mock startup initialization so tests don't load real PDFs or call Ollama.
with patch("app.rag_pipeline.initialize_rag", return_value=([], None, None, [])):
    from app.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_empty_question_is_rejected():
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_answerable_question_returns_answer_and_sources():
    expected = {
        "answer": "Air pollution can harm health and the environment.",
        "sources": [
            {
                "filename": "sample.pdf",
                "chunk_number": 1,
                "page_numbers": [1],
                "page_start": 1,
                "page_end": 1,
                "score": 0.03,
            }
        ],
    }

    with patch("app.api.ask_question", return_value=expected):
        response = client.post(
            "/ask",
            json={"question": "What are the effects of air pollution?"},
        )

    assert response.status_code == 200
    assert response.json() == expected
    assert response.json()["sources"][0]["filename"] == "sample.pdf"


def test_unsupported_question_returns_no_sources():
    expected = {
        "answer": "I couldn't find the answer in the provided documents.",
        "sources": [],
    }

    with patch("app.api.ask_question", return_value=expected):
        response = client.post(
            "/ask",
            json={"question": "What is the capital of Japan?"},
        )

    assert response.status_code == 200
    assert response.json() == expected
    assert response.json()["sources"] == []

def test_non_pdf_upload_is_rejected():
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"not a PDF", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Only PDF files are supported."
    }