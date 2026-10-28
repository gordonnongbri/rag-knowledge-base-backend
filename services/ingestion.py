from functools import lru_cache

import pymupdf
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from sqlalchemy.orm import Session

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # outputs 384-dim vectors
CHUNK_WORDS = 200    # MiniLM truncates input at ~256 tokens, so keep chunks under that
OVERLAP_WORDS = 40


@lru_cache
def get_model() -> SentenceTransformer:
    # Loaded once on first use. The first run downloads ~90MB.
    return SentenceTransformer(EMBEDDING_MODEL)


def parse_pdf(file_bytes: bytes) -> str:
    """Extract raw text from PDF bytes."""
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    return "\n\n".join(page.get_text() for page in doc)


def chunk_text(
    raw: str,
    chunk_words: int = CHUNK_WORDS,
    overlap: int = OVERLAP_WORDS,
) -> list[str]:
    """
    Paragraph-aware chunking: pack whole paragraphs into a chunk until the
    word budget is hit, split oversized paragraphs, and carry an overlap
    into the next chunk so context isn't cut mid-thought.
    """
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]

    # Break any single paragraph that exceeds the budget on its own
    units: list[list[str]] = []
    for para in paragraphs:
        words = para.split()
        for i in range(0, len(words), chunk_words):
            units.append(words[i : i + chunk_words])

    chunks: list[str] = []
    current: list[str] = []

    for words in units:
        if current and len(current) + len(words) > chunk_words:
            chunks.append(" ".join(current))
            current = current[-overlap:]  # carry the tail forward
        current.extend(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings locally. Used for both chunks and queries."""
    vectors = get_model().encode(
        texts,
        normalize_embeddings=True,  # makes cosine distance behave well
        batch_size=32,
        show_progress_bar=False,
    )
    return vectors.tolist()


def ingest_document(filename: str, file_bytes: bytes, db: Session) -> dict:
    """Full pipeline: parse -> chunk -> embed -> store."""
    raw_text = parse_pdf(file_bytes)
    chunks = chunk_text(raw_text)

    if not chunks:
        raise ValueError(
            "No extractable text found. The PDF may be a scanned image."
        )

    embeddings = embed_texts(chunks)

    doc_id = db.execute(
        text("INSERT INTO documents (filename) VALUES (:f) RETURNING id"),
        {"f": filename},
    ).scalar_one()

    db.execute(
        text(
            """
            INSERT INTO chunks (document_id, content, chunk_index, embedding)
            VALUES (:doc_id, :content, :idx, CAST(:emb AS vector))
            """
        ),
        [
            {"doc_id": doc_id, "content": c, "idx": i, "emb": str(e)}
            for i, (c, e) in enumerate(zip(chunks, embeddings))
        ],
    )

    db.commit()
    return {"document_id": doc_id, "chunks": len(chunks), "filename": filename}