import re

from app.pdf_loader import load_all_pdfs
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []

    # Find every page marker and its character position.
    page_markers = [
        (match.start(), int(match.group(1)))
        for match in re.finditer(r"--- PAGE (\d+) ---", text)
    ]

    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:

            # Find the page containing the beginning of the chunk.
            start_page = None

            for position, page_number in page_markers:
                if position <= start:
                    start_page = page_number
                else:
                    break

            # Find the page containing the end of the chunk.
            end_position = max(start, end - 1)

            end_page = None

            for position, page_number in page_markers:
                if position <= end_position:
                    end_page = page_number
                else:
                    break

            # If page information exists, include every page
            # touched by this chunk.
            if start_page is not None and end_page is not None:
                page_numbers = list(
                    range(start_page, end_page + 1)
                )

            elif start_page is not None:
                page_numbers = [start_page]

            elif end_page is not None:
                page_numbers = [end_page]

            else:
                page_numbers = []

            chunks.append({
                "text": chunk,
                "page_numbers": page_numbers
            })

        start += chunk_size - overlap

    return chunks


def load_all_chunks():

    documents = load_all_pdfs()

    all_chunks = []

    for document in documents:

        filename = document["filename"]
        text = document["text"]

        chunks = split_text(text)

        for chunk_number, chunk in enumerate(chunks, start=1):

            page_numbers = chunk["page_numbers"]

            all_chunks.append({
                "filename": filename,
                "chunk_number": chunk_number,
                "page_numbers": page_numbers,
                "page_start": min(page_numbers) if page_numbers else None,
                "page_end": max(page_numbers) if page_numbers else None,
                "text": chunk["text"]
            })

    return all_chunks