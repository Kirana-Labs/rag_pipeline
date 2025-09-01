import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional, List
import torch
import logging

from ..core.pipeline import RAGPipeline
from ..core.task_manager import BulkIngestionTaskManager
from .models import (
    IngestionRequest, IngestionResponse, QueryRequest as APIQueryRequest,
    QueryResponse as APIQueryResponse, QueryResult, HealthResponse,
    BulkIngestionRequest, BulkIngestionResponse, BulkJobStatus, JobStatus
)
from ..models.query import QueryRequest as PipelineQueryRequest
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
pipeline: Optional[RAGPipeline] = None
task_manager: Optional[BulkIngestionTaskManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global pipeline, task_manager
    
    database_url = os.getenv(
        "DATABASE_URL", 
        "postgresql://user:password@localhost:5432/rag_db"
    )
    use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dimensions = os.getenv("EMBEDDING_DIMENSIONS")
    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    use_reranker = os.getenv("USE_RERANKER", "false").lower() == "true"
    reranker_model = os.getenv("RERANKER_MODEL", "rerank-2.5")
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    max_concurrent_jobs = int(os.getenv("MAX_CONCURRENT_BULK_JOBS", "3"))
    max_concurrent_docs_per_job = int(os.getenv("MAX_CONCURRENT_DOCS_PER_JOB", "5"))
    
    # Convert embedding_dimensions to int if provided
    if embedding_dimensions:
        embedding_dimensions = int(embedding_dimensions)
    
    pipeline = RAGPipeline(
        database_url=database_url,
        use_gpu=use_gpu,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        voyage_api_key=voyage_api_key,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_reranker=use_reranker,
        reranker_model=reranker_model
    )
    
    task_manager = BulkIngestionTaskManager(
        max_concurrent_jobs=max_concurrent_jobs,
        max_concurrent_docs_per_job=max_concurrent_docs_per_job
    )
    
    try:
        await pipeline.initialize()
        logger.info("RAG Pipeline initialized successfully")
        logger.info("Bulk ingestion task manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise
    
    yield
    
    # Shutdown
    if task_manager:
        await task_manager.shutdown()
        logger.info("Task manager shut down")
    
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


def get_task_manager() -> BulkIngestionTaskManager:
    if task_manager is None:
        raise HTTPException(status_code=503, detail="Task manager not initialized")
    return task_manager


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
                end_char=result.get("end_char"),
                rank=result.get("rank"),
                relevance_score=result.get("relevance_score"),
                reranked=result.get("reranked", False)
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


@app.post("/ingest/bulk", response_model=BulkIngestionResponse)
async def bulk_ingest_documents(
    request: BulkIngestionRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    task_manager: BulkIngestionTaskManager = Depends(get_task_manager)
):
    try:
        logger.info(f"Starting bulk ingestion of {len(request.documents)} documents")
        
        # Create the bulk ingestion job
        job_id = task_manager.create_job(
            documents=request.documents,
            batch_name=request.batch_name
        )
        
        # Start processing in the background
        await task_manager.start_job(job_id, pipeline.ingest_document)
        
        # Estimate completion time (rough estimate: 30 seconds per document)
        estimated_minutes = max(1, (len(request.documents) * 30) // 60)
        estimated_time = f"{estimated_minutes}-{estimated_minutes + 2} minutes"
        
        return BulkIngestionResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            total_documents=len(request.documents),
            message="Bulk ingestion job created and processing started",
            estimated_completion_time=estimated_time
        )
    
    except Exception as e:
        logger.error(f"Bulk ingestion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start bulk ingestion: {str(e)}"
        )


@app.get("/ingest/bulk/{job_id}", response_model=BulkJobStatus)
async def get_bulk_job_status(
    job_id: str,
    task_manager: BulkIngestionTaskManager = Depends(get_task_manager)
):
    try:
        job_status = task_manager.get_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return job_status
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get job status failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job status: {str(e)}"
        )


@app.get("/ingest/bulk", response_model=List[BulkJobStatus])
async def list_bulk_jobs(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of jobs to return"),
    task_manager: BulkIngestionTaskManager = Depends(get_task_manager)
):
    try:
        jobs = task_manager.list_jobs(limit=limit)
        return jobs
    
    except Exception as e:
        logger.error(f"List jobs failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list jobs: {str(e)}"
        )


@app.delete("/ingest/bulk/{job_id}")
async def cancel_bulk_job(
    job_id: str,
    task_manager: BulkIngestionTaskManager = Depends(get_task_manager)
):
    try:
        cancelled = task_manager.cancel_job(job_id)
        if not cancelled:
            raise HTTPException(
                status_code=404, 
                detail="Job not found or not running"
            )
        
        return {"message": f"Job {job_id} cancelled successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cancel job failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel job: {str(e)}"
        )


@app.post("/ingest/bulk/cleanup")
async def cleanup_old_jobs(
    max_age_hours: int = Query(24, ge=1, le=720, description="Maximum age in hours for jobs to keep"),
    task_manager: BulkIngestionTaskManager = Depends(get_task_manager)
):
    try:
        cleaned_count = task_manager.cleanup_completed_jobs(max_age_hours=max_age_hours)
        return {
            "message": f"Cleaned up {cleaned_count} old jobs",
            "cleaned_count": cleaned_count
        }
    
    except Exception as e:
        logger.error(f"Cleanup jobs failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup jobs: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)