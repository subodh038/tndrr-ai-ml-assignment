"""
ingestion.py - Document loading and text extraction for the RAG pipeline.
Supports PDF files (via PyMuPDF) and plain text files.
"""
import os
import re
from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    text: str
    source: str
    page: int


def extract_pdf(pdf_path: str) -> List[Document]:
    """Extract text from a PDF file, one Document per page."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    docs = []
    try:
        pdf = fitz.open(pdf_path)
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text()
            text = clean_text(text)
            if text.strip():
                docs.append(Document(
                    text=text,
                    source=os.path.basename(pdf_path),
                    page=page_num + 1
                ))
        pdf.close()
    except Exception as e:
        print(f"[ingestion] Failed to extract {pdf_path}: {e}")
    return docs


def load_text_file(path: str) -> List[Document]:
    """Load a plain text file, splitting into ~500-char pseudo-pages."""
    docs = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        raw = clean_text(raw)
        # Split into ~500-char chunks treated as "pages"
        chunk_size = 500
        source = os.path.basename(path)
        for i, start in enumerate(range(0, len(raw), chunk_size)):
            chunk_text = raw[start:start + chunk_size].strip()
            if chunk_text:
                docs.append(Document(
                    text=chunk_text,
                    source=source,
                    page=i + 1
                ))
    except Exception as e:
        print(f"[ingestion] Failed to load {path}: {e}")
    return docs


def clean_text(text: str) -> str:
    """
    Clean extracted text:
    - Fix hyphenation at line breaks (e.g., "trans-\nformer" -> "transformer")
    - Normalize whitespace and remove control characters
    - Remove page artifacts like lone numbers, repeated dashes
    """
    # Fix hyphenated line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Remove control characters except newlines/tabs
    text = re.sub(r"[^\S\n\t ]+", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are just numbers (page numbers)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    # Remove lines of dashes/underscores (artifacts)
    text = re.sub(r"^[-_=]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def ingest_directory(data_dir: str) -> List[Document]:
    """
    Scan a directory for .pdf and .txt files, extract all documents.
    Returns a flat list of Document objects sorted by source + page.
    """
    all_docs = []
    supported = {".pdf": extract_pdf, ".txt": load_text_file}

    if not os.path.isdir(data_dir):
        raise ValueError(f"Not a directory: {data_dir}")

    files = sorted(os.listdir(data_dir))
    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        if ext in supported:
            full_path = os.path.join(data_dir, fname)
            print(f"[ingestion] Loading {fname} ...")
            docs = supported[ext](full_path)
            print(f"[ingestion]   -> {len(docs)} segments")
            all_docs.extend(docs)

    print(f"[ingestion] Total documents loaded: {len(all_docs)}")
    return all_docs


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    docs = ingest_directory(data_dir)
    print(f"Loaded {len(docs)} documents")
    if docs:
        print("First doc preview:", docs[0].text[:200])
