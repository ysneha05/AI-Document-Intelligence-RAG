from pathlib import Path

from sentence_transformers import SentenceTransformer

from chunker import split_text
from pdf_loader import extract_text


PDF_PATH = Path("data/sample.pdf")

MODEL_NAME = "all-MiniLM-L6-v2"


def create_embeddings(chunks):
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")
    embeddings = model.encode(
        chunks,
        show_progress_bar=True
    )

    return model, embeddings


if __name__ == "__main__":
    print("Loading document...")

    text = extract_text(PDF_PATH)
    chunks = split_text(text)

    print(f"Total chunks: {len(chunks)}")

    model, embeddings = create_embeddings(chunks)

    print("\nEmbedding generation successful!")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Embedding data type: {embeddings.dtype}")

    print("\nFirst embedding:")
    print(embeddings[0])