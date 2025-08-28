from typing import Dict, Any, Optional
from pydantic import BaseModel, HttpUrl, Field


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


class HealthResponse(BaseModel):
    status: str
    version: str
    database_status: str
    gpu_available: bool