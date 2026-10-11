import time
from app.semantic_index import build_index
from app.rag_pipeline import ask_question, initialize_rag
from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.config import DATA_DIR
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

    if not file.filename.lower().endswith(".pdf"):
        return {
            "error": "Only PDF files are supported."
        }

    file_path = DATA_DIR / file.filename

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Rebuild semantic embeddings for all documents
    build_index()

    # Reload the complete RAG indexes
    chunks, vectorizer, matrix, semantic_index = initialize_rag()

    return {
        "message": "PDF uploaded and RAG indexes rebuilt successfully.",
        "filename": file.filename,
        "total_chunks": len(chunks)
    }