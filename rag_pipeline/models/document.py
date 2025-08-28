from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import uuid


class DocumentMetadata(BaseModel):
    source_url: str
    filename: str
    file_type: str
    file_size: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None
    
    class Config:
        arbitrary_types_allowed = True


class DocumentChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str
    chunk_index: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    embedding: Optional[List[float]] = None
    chunk_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True