from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ..db import get_db
from ..services.retrieval import retrieve_chunks
from ..services.generation import stream_answer
import json

router = APIRouter()

class QueryRequest(BaseModel):  
    question: str
    top_k: int = 5

@router.post("/query")
async def query_knowledge_base(request: QueryRequest, db = Depends(get_db)):
    chunks = retrieve_chunks(request.question, db, top_k=request.top_k)
    
    async def event_stream():

        yield f"data: {json.dumps({'type': 'sources', 'chunks': chunks})}\n\n"
        
        async for token in stream_answer(request.question, chunks):
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no" 
        }
    )