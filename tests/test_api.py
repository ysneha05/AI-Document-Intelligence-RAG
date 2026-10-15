
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

def test_upload_returns_error_when_indexing_fails(tmp_path):
    from app import api

    with patch.object(api, "DATA_DIR", tmp_path):
        with patch.object(
            api,
            "build_index",
            side_effect=RuntimeError("Indexing failed"),
        ):
            response = client.post(
                "/upload",
                files={
                    "file": (
                        "test.pdf",
                        b"%PDF-1.4 test content",
                        "application/pdf",
                    )
                },
            )

    assert response.status_code == 500


def test_build_index_preserves_existing_index_if_save_fails(tmp_path):
    import json
    from unittest.mock import patch

    from app import semantic_index

    index_path = tmp_path / "semantic_index.json"
    index_path.write_text('[{"old": true}]', encoding="utf-8")

    with (
        patch.object(semantic_index, "INDEX_PATH", index_path),
        patch.object(
            semantic_index,
            "load_all_chunks",
            return_value=[{
                "text": "test text",
                "filename": "test.pdf",
                "chunk_number": 1,
                "page_numbers": [1],
            }],
        ),
        patch.object(
            semantic_index,
            "create_embedding",
            return_value=[0.1, 0.2],
        ),
        patch("app.semantic_index.json.dump", side_effect=OSError("Disk write failed")),
    ):
        try:
            semantic_index.build_index()
            assert False, "Expected an OSError"
        except OSError:
            pass

    assert json.loads(index_path.read_text(encoding="utf-8")) == [{"old": True}]
    assert not (tmp_path / "semantic_index.tmp").exists()


def test_failed_upload_removes_new_pdf_and_restores_index(tmp_path):
    from app import api

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    index_path = data_dir / "semantic_index.json"
    index_path.write_text('[{"old": true}]', encoding="utf-8")

    with (
        patch.object(api, "DATA_DIR", data_dir),
        patch.object(api, "SEMANTIC_INDEX_PATH", index_path),
        patch.object(
            api,
            "build_index",
            side_effect=RuntimeError("Indexing failed"),
        ),
    ):
        response = client.post(
            "/upload",
            files={
                "file": (
                    "new.pdf",
                    b"%PDF-1.4 new content",
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 500
    assert not (data_dir / "new.pdf").exists()
    assert index_path.read_text(encoding="utf-8") == '[{"old": true}]'


def test_failed_replacement_restores_existing_pdf_and_index(tmp_path):
    from app import api

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    existing_pdf = data_dir / "existing.pdf"
    existing_pdf.write_bytes(b"original PDF content")

    index_path = data_dir / "semantic_index.json"
    index_path.write_text('[{"old": true}]', encoding="utf-8")

    def replace_index_then_fail():
        index_path.write_text('[{"new": true}]', encoding="utf-8")
        raise RuntimeError("Initialization failed")

    with (
        patch.object(api, "DATA_DIR", data_dir),
        patch.object(api, "SEMANTIC_INDEX_PATH", index_path),
        patch.object(api, "build_index", side_effect=replace_index_then_fail),
    ):
        response = client.post(
            "/upload",
            files={
                "file": (
                    "existing.pdf",
                    b"replacement PDF content",
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 500
    assert existing_pdf.read_bytes() == b"original PDF content"
    assert index_path.read_text(encoding="utf-8") == '[{"old": true}]'
