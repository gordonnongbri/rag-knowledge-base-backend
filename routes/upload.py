from fastapi import APIRouter, UploadFile, File, Depends
from ..db import get_db
from ..services.ingestion import ingest_document

router = APIRouter()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db = Depends(get_db)
):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are supported"}
    
    file_bytes = await file.read()
    result = ingest_document(file.filename, file_bytes, db)
    return {"success": True, **result}