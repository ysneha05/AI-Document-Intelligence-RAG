from pathlib import Path
from pypdf import PdfReader

DATA_DIR = Path("data")


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)

    all_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        # Clean common PDF extraction artifacts
        text = text.replace("-\n", "")
        text = text.replace("\n", " ")
        text = " ".join(text.split())

        all_text.append(
            f"\n--- PAGE {page_number} ---\n{text}"
        )

    return "\n".join(all_text)


def load_all_pdfs():
    documents = []

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")

        text = extract_text(pdf_path)

        documents.append({
            "filename": pdf_path.name,
            "text": text
        })

    return documents


if __name__ == "__main__":
    documents = load_all_pdfs()

    print("\nPDF loading completed!")
    print(f"Total PDFs: {len(documents)}")

    for document in documents:
        print("\n" + "=" * 70)
        print(f"File: {document['filename']}")
        print(f"Characters extracted: {len(document['text'])}")