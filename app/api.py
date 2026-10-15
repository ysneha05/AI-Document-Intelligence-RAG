import tempfile
from pathlib import Path
import time
from app.semantic_index import build_index
from app.rag_pipeline import ask_question, initialize_rag
from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.config import DATA_DIR, SEMANTIC_INDEX_PATH
from pydantic import BaseModel, field_validator


app = FastAPI(
    title="AI Document Intelligence & RAG Assistant",
    description="A document question-answering API using hybrid retrieval and a local LLM.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chunks, vectorizer, matrix, semantic_index = initialize_rag()


class QuestionRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty.")

        return value


@app.get("/")
def root():
    return {
        "message": "AI Document Intelligence & RAG Assistant API is running."
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/ask")
def ask(request: QuestionRequest):
    try:
        request_timer = time.perf_counter()

        result = ask_question(
            request.question,
            chunks,
            vectorizer,
            matrix,
            semantic_index
        )

        print(
            "Total server-side /ask time:",
            round(time.perf_counter() - request_timer, 2),
            "seconds"
        )

        return result
    except Exception:
        return {
            "error": "An internal error occurred while processing the question."
        }
        



@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global chunks, vectorizer, matrix, semantic_index

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    filename = Path(file.filename).name
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DATA_DIR / filename

    old_file_bytes = file_path.read_bytes() if file_path.exists() else None
    old_index_bytes = (
        SEMANTIC_INDEX_PATH.read_bytes()
        if SEMANTIC_INDEX_PATH.exists()
        else None
    )

    temp_upload = None

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="The PDF is empty.")

        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".upload-",
            suffix=".pdf",
            dir=DATA_DIR,
            delete=False,
        ) as temp_file:
            temp_upload = Path(temp_file.name)
            temp_file.write(content)

        temp_upload.replace(file_path)

        build_index()

        new_chunks, new_vectorizer, new_matrix, new_semantic_index = (
            initialize_rag()
        )

        chunks = new_chunks
        vectorizer = new_vectorizer
        matrix = new_matrix
        semantic_index = new_semantic_index

        return {
            "message": "PDF uploaded and RAG indexes rebuilt successfully.",
            "filename": filename,
            "total_chunks": len(chunks)
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(f"PDF upload/indexing failed: {exc}")

        try:
            if old_file_bytes is None:
                file_path.unlink(missing_ok=True)
            else:
                file_path.write_bytes(old_file_bytes)

            if old_index_bytes is None:
                SEMANTIC_INDEX_PATH.unlink(missing_ok=True)
            else:
                SEMANTIC_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
                rollback_path = SEMANTIC_INDEX_PATH.with_suffix(".rollback")
                try:
                    rollback_path.write_bytes(old_index_bytes)
                    rollback_path.replace(SEMANTIC_INDEX_PATH)
                finally:
                    rollback_path.unlink(missing_ok=True)

        except Exception as rollback_exc:
            print(f"Upload rollback also failed: {rollback_exc}")

        raise HTTPException(
            status_code=500,
            detail="The PDF could not be processed. Check the server logs."
        ) from exc

    finally:
        if temp_upload is not None:
            temp_upload.unlink(missing_ok=True)
        await file.close()
