from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, JSON, Float, Index, text
from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

Base = declarative_base()


class DocumentRecord(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content = Column(Text, nullable=False)
    source_url = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    custom_metadata = Column(JSONB, default=dict)
    embedding = Column(Vector(384))  # Default for all-MiniLM-L6-v2
    
    __table_args__ = (
        Index('idx_documents_source_url', 'source_url'),
        Index('idx_documents_file_type', 'file_type'),
        Index('idx_documents_created_at', 'created_at'),
    )


class DocumentChunkRecord(Base):
    __tablename__ = "document_chunks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    start_char = Column(Integer)
    end_char = Column(Integer)
    embedding = Column(Vector(384))  # Default for all-MiniLM-L6-v2
    chunk_metadata = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_chunks_document_id', 'document_id'),
        Index('idx_chunks_chunk_index', 'chunk_index'),
        Index('idx_chunks_embedding', 'embedding', postgresql_using='ivfflat'),
    )


class DatabaseManager:
    def __init__(self, database_url: str, echo: bool = False):
        self.database_url = database_url
        self.engine = create_async_engine(database_url.replace("postgresql://", "postgresql+asyncpg://"), echo=echo)
        self.SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    async def initialize_database(self):
        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
    
    def get_session(self) -> AsyncSession:
        return self.SessionLocal()
    
    async def close(self):
        self.engine.dispose()