from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text, and_, or_
from sqlalchemy.future import select
import asyncio
from functools import partial

from .database import DatabaseManager, DocumentRecord, DocumentChunkRecord
from ..models.document import Document, DocumentChunk, DocumentMetadata


class VectorStore:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def store_document(self, document: Document) -> str:
        # loop = asyncio.get_event_loop()
        
        async def _store():
            async with self.db_manager.get_session() as session:
                doc_record = DocumentRecord(
                    id=document.id,
                    content=document.content,
                    source_url=document.metadata.source_url,
                    filename=document.metadata.filename,
                    file_type=document.metadata.file_type,
                    file_size=document.metadata.file_size,
                    custom_metadata=document.metadata.custom_metadata,
                    embedding=document.embedding
                )
                session.add(doc_record)
                await session.commit()
                return document.id
        
        # return await loop.run_in_executor(None, _store)
        return await _store()
    
    async def store_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        # loop = asyncio.get_event_loop()
        
        async def _store():
            async with self.db_manager.get_session() as session:
                chunk_records = []
                for chunk in chunks:
                    chunk_record = DocumentChunkRecord(
                        id=chunk.id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        chunk_index=chunk.chunk_index,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        embedding=chunk.embedding,
                        chunk_metadata=chunk.chunk_metadata
                    )
                    chunk_records.append(chunk_record)
                
                session.add_all(chunk_records)
                await session.commit()
                return [chunk.id for chunk in chunks]
        
        # return await loop.run_in_executor(None, _store)
        return await _store()
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[DocumentChunk, float]]:
        # loop = asyncio.get_event_loop()
        
        async def _search():
            async with self.db_manager.get_session() as session:
                # Base query with similarity search
                query = select(
                    DocumentChunkRecord,
                    DocumentChunkRecord.embedding.cosine_distance(query_embedding).label('distance')
                )
                
                # Apply metadata filters
                if metadata_filters:
                    conditions = []
                    for key, value in metadata_filters.items():
                        if isinstance(value, list):
                            # Handle list values (IN operator)
                            conditions.append(
                                DocumentChunkRecord.chunk_metadata[key].astext.in_(value)
                            )
                        else:
                            # Handle single values
                            conditions.append(
                                DocumentChunkRecord.chunk_metadata[key].astext == str(value)
                            )
                    
                    if conditions:
                        query = query.filter(and_(*conditions))
                
                # Apply similarity threshold (convert to distance threshold)
                distance_threshold = 1.0 - similarity_threshold
                query = query.filter(
                    DocumentChunkRecord.embedding.cosine_distance(query_embedding) < distance_threshold
                )
                
                # Order by similarity and limit
                results = await session.execute(query.order_by('distance').limit(top_k))
                
                # Convert to DocumentChunk objects with similarity scores
                chunks_with_scores = []
                for record, distance in results:
                    similarity = 1.0 - distance
                    chunk = DocumentChunk(
                        id=record.id,
                        document_id=record.document_id,
                        content=record.content,
                        chunk_index=record.chunk_index,
                        start_char=record.start_char,
                        end_char=record.end_char,
                        embedding=record.embedding,
                        chunk_metadata=record.chunk_metadata
                    )
                    chunks_with_scores.append((chunk, similarity))
                
                return chunks_with_scores
        
        # return await loop.run_in_executor(None, _search)
        return await _search()
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        # loop = asyncio.get_event_loop()
        
        async def _get():
            async with self.db_manager.get_session() as session:
                record = await session.execute(
                    select(DocumentRecord).filter(
                        DocumentRecord.id == document_id
                    ).first()
                )
                
                if not record:
                    return None
                
                metadata = DocumentMetadata(
                    source_url=record.source_url,
                    filename=record.filename,
                    file_type=record.file_type,
                    file_size=record.file_size,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    custom_metadata=record.custom_metadata
                )
                
                return Document(
                    id=record.id,
                    content=record.content,
                    metadata=metadata,
                    embedding=record.embedding
                )
        
        # return await loop.run_in_executor(None, _get)
        return await _get()
    
    async def delete_document(self, document_id: str) -> bool:
        # loop = asyncio.get_event_loop()
        
        async def _delete():
            async with self.db_manager.get_session() as session:
                # Delete chunks first
                await session.query(DocumentChunkRecord).filter(
                    DocumentChunkRecord.document_id == document_id
                ).delete()
                
                # Delete document
                deleted = await session.query(DocumentRecord).filter(
                    DocumentRecord.id == document_id
                ).delete()
                
                await session.commit()
                return deleted > 0
        
        # return await loop.run_in_executor(None, _delete)
        return await _delete()
    
    async def get_documents_by_metadata(
        self, 
        metadata_filters: Dict[str, Any],
        limit: int = 100
    ) -> List[Document]:
        # loop = asyncio.get_event_loop()
        
        async def _get():
            async with self.db_manager.get_session() as session:
                query = select(DocumentRecord)
                
                conditions = []
                for key, value in metadata_filters.items():
                    if key in ['source_url', 'filename', 'file_type']:
                        # Direct column filters
                        column = getattr(DocumentRecord, key)
                        if isinstance(value, list):
                            conditions.append(column.in_(value))
                        else:
                            conditions.append(column == value)
                    else:
                        # JSON metadata filters
                        if isinstance(value, list):
                            conditions.append(
                                DocumentRecord.custom_metadata[key].astext.in_(value)
                            )
                        else:
                            conditions.append(
                                DocumentRecord.custom_metadata[key].astext == str(value)
                            )
                
                if conditions:
                    query = query.filter(and_(*conditions))
                
                records = await session.execute(query.limit(limit).all())
                
                documents = []
                for record in records:
                    metadata = DocumentMetadata(
                        source_url=record.source_url,
                        filename=record.filename,
                        file_type=record.file_type,
                        file_size=record.file_size,
                        created_at=record.created_at,
                        updated_at=record.updated_at,
                        custom_metadata=record.custom_metadata
                    )
                    
                    document = Document(
                        id=record.id,
                        content=record.content,
                        metadata=metadata,
                        embedding=record.embedding
                    )
                    documents.append(document)
                
                return documents
        
        # return await loop.run_in_executor(None, _get)
        return await _get()