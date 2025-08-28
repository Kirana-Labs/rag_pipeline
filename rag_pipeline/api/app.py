import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional, List
import torch
import logging

from ..core.pipeline import RAGPipeline
from .models import (
    IngestionRequest, IngestionResponse, QueryRequest as APIQueryRequest,
    QueryResponse as APIQueryResponse, QueryResult, HealthResponse
)
from ..models.query import QueryRequest as PipelineQueryRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global pipeline
    
    database_url = os.getenv(
        "DATABASE_URL", 
        "postgresql://user:password@localhost:5432/rag_db"
    )
    use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    pipeline = RAGPipeline(
        database_url=database_url,
        use_gpu=use_gpu,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    try:
        await pipeline.initialize()
        logger.info("RAG Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise
    
    yield
    
    # Shutdown
    if pipeline:
        await pipeline.close()
        logger.info("RAG Pipeline closed")


app = FastAPI(
    title="RAG Pipeline API",
    description="A GPU-accelerated RAG pipeline using Docling and LangChain",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline() -> RAGPipeline:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        # Check database connection
        db_status = "connected"  # You might want to add an actual DB ping here
        gpu_available = torch.cuda.is_available()
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            database_status=db_status,
            gpu_available=gpu_available
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            database_status="error",
            gpu_available=False
        )


@app.post("/ingest", response_model=IngestionResponse)
async def ingest_document(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    pipeline: RAGPipeline = Depends(get_pipeline)
):
    try:
        logger.info(f"Starting ingestion for {request.filename}")
        
        document_id = await pipeline.ingest_document(
            url=str(request.url),
            filename=request.filename,
            custom_metadata=request.metadata
        )
        
        logger.info(f"Successfully ingested document {document_id}")
        
        return IngestionResponse(
            document_id=document_id,
            status="success",
            message="Document ingested successfully"
        )
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest document: {str(e)}"
        )


@app.post("/query", response_model=APIQueryResponse)
async def query_documents(
    request: APIQueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline)
):
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Convert API request to pipeline request
        pipeline_request = PipelineQueryRequest(
            query=request.query,
            top_k=request.top_k,
            metadata_filters=request.metadata_filters,
            similarity_threshold=request.similarity_threshold
        )
        
        response = await pipeline.query_documents(pipeline_request)
        
        # Convert pipeline response to API response
        api_results = [
            QueryResult(
                id=result["id"],
                document_id=result["document_id"],
                content=result["content"],
                similarity_score=result["similarity_score"],
                metadata=result["metadata"],
                chunk_index=result["chunk_index"],
                start_char=result.get("start_char"),
                end_char=result.get("end_char")
            )
            for result in response.results
        ]
        
        return APIQueryResponse(
            query=response.query,
            results=api_results,
            total_results=response.total_results,
            execution_time_ms=response.execution_time_ms
        )
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@app.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    pipeline: RAGPipeline = Depends(get_pipeline)
):
    try:
        document = await pipeline.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "id": document.id,
            "content": document.content,
            "metadata": {
                "source_url": document.metadata.source_url,
                "filename": document.metadata.filename,
                "file_type": document.metadata.file_type,
                "file_size": document.metadata.file_size,
                "created_at": document.metadata.created_at,
                "updated_at": document.metadata.updated_at,
                "custom_metadata": document.metadata.custom_metadata
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve document: {str(e)}"
        )


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    pipeline: RAGPipeline = Depends(get_pipeline)
):
    try:
        deleted = await pipeline.delete_document(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": "Document deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )


@app.get("/documents")
async def list_documents(
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of documents to return"),
    pipeline: RAGPipeline = Depends(get_pipeline)
):
    try:
        metadata_filters = {}
        if file_type:
            metadata_filters["file_type"] = file_type
        if category:
            metadata_filters["category"] = category
        
        documents = await pipeline.get_documents_by_metadata(
            metadata_filters=metadata_filters,
            limit=limit
        )
        
        return {
            "documents": [
                {
                    "id": doc.id,
                    "metadata": {
                        "source_url": doc.metadata.source_url,
                        "filename": doc.metadata.filename,
                        "file_type": doc.metadata.file_type,
                        "file_size": doc.metadata.file_size,
                        "created_at": doc.metadata.created_at,
                        "updated_at": doc.metadata.updated_at,
                        "custom_metadata": doc.metadata.custom_metadata
                    }
                }
                for doc in documents
            ],
            "total": len(documents)
        }
    
    except Exception as e:
        logger.error(f"List documents failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)