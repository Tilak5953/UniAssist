import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path

from config import (
    KNOWLEDGE_BASE_DIR,
    INDEX_STORAGE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_THRESHOLD,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIM
)
from chunking import DocumentChunker
from embeddings import EmbeddingEngine
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_service")

app = FastAPI(
    title="UniAssist RAG Service",
    description="Knowledge Base Ingestion, Chunking, Embedding & Vector Similarity Search",
    version="1.0.0"
)

# Initialize components
chunker = DocumentChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
embedder = EmbeddingEngine(ollama_url=OLLAMA_BASE_URL, model_name=EMBEDDING_MODEL, dim=EMBEDDING_DIM)
vector_store = VectorStore(storage_path=INDEX_STORAGE_PATH)

# Request / Response Schemas
class RetrieveRequest(BaseModel):
    query: str = Field(..., description="The query string to search for")
    top_k: Optional[int] = Field(default=DEFAULT_TOP_K, description="Number of relevant chunks to retrieve")

class RetrievedChunk(BaseModel):
    chunk_id: str
    source_file: str
    document_title: str
    section_title: str
    text: str
    raw_text: str
    similarity_score: float
    vector_score: float

class RetrieveResponse(BaseModel):
    query: str
    total_retrieved: int
    results: List[RetrievedChunk]

class IngestResponse(BaseModel):
    status: str
    files_processed: int
    total_chunks: int
    storage_path: str

@app.on_event("startup")
def startup_event():
    """Load existing index or ingest knowledge base on startup."""
    logger.info("Initializing RAG service...")
    loaded = vector_store.load_from_disk()
    if loaded and len(vector_store.chunks) > 0:
        logger.info(f"Loaded existing index with {len(vector_store.chunks)} chunks.")
    else:
        logger.info("No existing index found. Ingesting knowledge base documents...")
        ingest_documents()

def ingest_documents() -> IngestResponse:
    kb_path = Path(KNOWLEDGE_BASE_DIR)
    if not kb_path.exists():
        logger.warning(f"Knowledge base directory does not exist: {kb_path}")
        return IngestResponse(status="error", files_processed=0, total_chunks=0, storage_path=str(INDEX_STORAGE_PATH))

    logger.info(f"Ingesting documents from {kb_path}...")
    chunks = chunker.process_directory(kb_path)
    if not chunks:
        logger.warning("No markdown/text files found to chunk.")
        return IngestResponse(status="empty", files_processed=0, total_chunks=0, storage_path=str(INDEX_STORAGE_PATH))

    logger.info(f"Generated {len(chunks)} chunks. Generating embeddings...")
    texts = [c["text"] for c in chunks]
    vectors = embedder.get_batch_embeddings(texts)

    vector_store.clear()
    vector_store.add_items(chunks, vectors)
    vector_store.save_to_disk()
    logger.info(f"Successfully saved {len(chunks)} chunks to {INDEX_STORAGE_PATH}")

    files = set(c["source_file"] for c in chunks)
    return IngestResponse(
        status="success",
        files_processed=len(files),
        total_chunks=len(chunks),
        storage_path=str(INDEX_STORAGE_PATH)
    )

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "rag_service",
        "chunks_indexed": len(vector_store.chunks)
    }

@app.post("/ingest", response_model=IngestResponse)
def trigger_ingest():
    """Manual re-indexing endpoint."""
    return ingest_documents()

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    """Retrieve top-K relevant chunks for a user query."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_vec = embedder.get_embedding(req.query)
    top_k = req.top_k if req.top_k and req.top_k > 0 else DEFAULT_TOP_K
    results = vector_store.search(
        query=req.query,
        query_vector=query_vec,
        top_k=top_k,
        min_score=MIN_SIMILARITY_THRESHOLD
    )

    return RetrieveResponse(
        query=req.query,
        total_retrieved=len(results),
        results=results
    )

@app.get("/documents")
def list_documents():
    """List all indexed documents with metadata and chunk counts."""
    file_map = {}
    for c in vector_store.chunks:
        fn = c["source_file"]
        if fn not in file_map:
            file_map[fn] = {
                "source_file": fn,
                "document_title": c["document_title"],
                "chunks_count": 0,
                "sections": set()
            }
        file_map[fn]["chunks_count"] += 1
        file_map[fn]["sections"].add(c["section_title"])

    result = []
    for fn, data in file_map.items():
        result.append({
            "source_file": data["source_file"],
            "document_title": data["document_title"],
            "chunks_count": data["chunks_count"],
            "sections": sorted(list(data["sections"]))
        })
    return {"documents": result, "total_documents": len(result), "total_chunks": len(vector_store.chunks)}

@app.get("/stats")
def get_stats():
    return {
        "total_documents": len(set(c["source_file"] for c in vector_store.chunks)),
        "total_chunks": len(vector_store.chunks),
        "embedding_dim": EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "knowledge_base_dir": str(KNOWLEDGE_BASE_DIR),
        "index_storage_path": str(INDEX_STORAGE_PATH)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
