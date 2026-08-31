from sqlalchemy.orm import Session
from sqlalchemy import text
from .ingestion import embed_texts

def retrieve_chunks(query: str, db: Session, top_k: int = 5) -> list[dict]:
    """Embed the query and find the top-k most similar chunks."""
    query_embedding = embed_texts([query])[0]
    
    results = db.execute(
        text("""
            SELECT content, chunk_index,
                   1 - (embedding <=> :emb) AS similarity
            FROM chunks
            ORDER BY embedding <=> :emb
            LIMIT :k
        """),
        {"emb": str(query_embedding), "k": top_k}
    ).fetchall()
    
    return [
        {"content": r[0], "chunk_index": r[1], "similarity": float(r[2])}
        for r in results
    ]