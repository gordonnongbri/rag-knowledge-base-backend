from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .db import init_db
from .routes import upload, query
from .services.ingestion import get_model

app = FastAPI(title="RAG Knowledge Base API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
    
@app.on_event("startup")
def startup():
    init_db()
    get_model()

app.include_router(upload.router, prefix="/api")
app.include_router(query.router, prefix="/api")