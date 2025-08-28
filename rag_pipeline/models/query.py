from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    metadata_filters: Optional[Dict[str, Any]] = None
    similarity_threshold: float = 0.7


class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    execution_time_ms: float