import fitz  # PyMuPDF
import openai
import os
from sqlalchemy.orm import Session
from sqlalchemy import text

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_pdf(file_bytes: bytes) -> str:
    """Extract raw text from PDF bytes."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Semantic chunking: split on paragraphs first,
    then enforce token budget with overlap for context continuity.
    """
    # Split on double newlines (paragraph boundaries) first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        words = para.split()
        if current_len + len(words) > chunk_size:
            if current:
                chunks.append(" ".join(current))
            # Start next chunk with overlap from end of previous
            current = current[-overlap:] + words
            current_len = len(current)
        else:
            current.extend(words)
            current_len += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed text chunks using OpenAI embeddings."""
    response = client.embeddings.create(
        model="text-embedding-3-small",  # 1536 dims, cheap and fast
        input=texts
    )
    return [item.embedding for item in response.data]

def ingest_document(filename: str, file_bytes: bytes, db: Session) -> dict:
    """Full ingestion pipeline: parse → chunk → embed → store."""
    # 1. Parse
    raw_text = parse_pdf(file_bytes)
    
    # 2. Chunk
    chunks = chunk_text(raw_text)
    
    # 3. Embed (batch for efficiency)
    embeddings = embed_texts(chunks)
    
    # 4. Store document record
    result = db.execute(
        text("INSERT INTO documents (filename) VALUES (:f) RETURNING id"),
        {"f": filename}
    )
    doc_id = result.fetchone()[0]
    
    # 5. Store chunks + embeddings
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        db.execute(
            text("""
                INSERT INTO chunks (document_id, content, chunk_index, embedding)
                VALUES (:doc_id, :content, :idx, :emb)
            """),
            {
                "doc_id": doc_id,
                "content": chunk,
                "idx": i,
                "emb": str(embedding)  # pgvector accepts list as string
            }
        )
    
    db.commit()
    return {"document_id": doc_id, "chunks": len(chunks), "filename": filename}