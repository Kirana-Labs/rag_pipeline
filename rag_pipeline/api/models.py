from typing import Dict, Any, Optional, List
from pydantic import BaseModel, HttpUrl, Field
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentIngestionItem(BaseModel):
    url: HttpUrl
    filename: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/document.pdf",
                "filename": "example_document.pdf", 
                "metadata": {
                    "category": "research",
                    "author": "John Doe"
                }
            }
        }


class IngestionRequest(BaseModel):
    url: HttpUrl
    filename: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/document.pdf",
                "filename": "example_document.pdf",
                "metadata": {
                    "category": "research",
                    "author": "John Doe",
                    "tags": ["AI", "ML"]
                }
            }
        }


class IngestionResponse(BaseModel):
    document_id: str
    status: str
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "success",
                "message": "Document ingested successfully"
            }
        }


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    metadata_filters: Optional[Dict[str, Any]] = None
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "top_k": 5,
                "metadata_filters": {
                    "category": "research",
                    "tags": ["AI"]
                },
                "similarity_threshold": 0.7
            }
        }


class QueryResult(BaseModel):
    id: str
    document_id: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    chunk_index: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class QueryResponse(BaseModel):
    query: str
    results: list[QueryResult]
    total_results: int
    execution_time_ms: float


class BulkIngestionRequest(BaseModel):
    documents: List[DocumentIngestionItem] = Field(min_items=1, max_items=100)
    batch_name: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "url": "https://example.com/doc1.pdf",
                        "filename": "document1.pdf",
                        "metadata": {"category": "research", "priority": "high"}
                    },
                    {
                        "url": "https://example.com/doc2.pdf", 
                        "filename": "document2.pdf",
                        "metadata": {"category": "research", "priority": "medium"}
                    }
                ],
                "batch_name": "research_papers_batch_1"
            }
        }


class BulkIngestionResponse(BaseModel):
    job_id: str
    status: JobStatus
    total_documents: int
    message: str
    estimated_completion_time: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "bulk_123e4567-e89b-12d3-a456-426614174000",
                "status": "pending",
                "total_documents": 50,
                "message": "Bulk ingestion job created successfully",
                "estimated_completion_time": "5-10 minutes"
            }
        }


class DocumentIngestionResult(BaseModel):
    url: str
    filename: str
    status: JobStatus
    document_id: Optional[str] = None
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None


class BulkJobStatus(BaseModel):
    job_id: str
    status: JobStatus
    batch_name: Optional[str] = None
    total_documents: int
    processed_documents: int
    successful_documents: int
    failed_documents: int
    results: List[DocumentIngestionResult]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    database_status: str
    gpu_available: bool