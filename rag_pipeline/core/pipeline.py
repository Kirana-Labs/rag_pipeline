import time
from typing import Dict, Any, List, Optional, Tuple

from .document_processor import DocumentProcessor
from .chunker import ChunkingService
from .embedder import EmbeddingService
from ..storage.vector_store import VectorStore
from ..storage.database import DatabaseManager
from ..models.document import Document, DocumentChunk
from ..models.query import QueryRequest, QueryResponse


class RAGPipeline:
    def __init__(
        self,
        database_url: str,
        use_gpu: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.database_url = database_url
        self.use_gpu = use_gpu
        
        # Initialize components
        self.db_manager = DatabaseManager(database_url)
        self.vector_store = VectorStore(self.db_manager)
        self.document_processor = DocumentProcessor(use_gpu=use_gpu)
        self.chunking_service = ChunkingService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedding_service = EmbeddingService(
            model_name=embedding_model,
            device="cuda" if use_gpu else "mps"
        )
    
    async def initialize(self):
        await self.db_manager.initialize_database()
    
    async def ingest_document(
        self,
        url: str,
        filename: str,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        # Process document
        document = await self.document_processor.process_document_from_url(
            url=url,
            filename=filename,
            custom_metadata=custom_metadata
        )
        
        # Chunk document
        chunks = await self.chunking_service.chunk_document(document)
        
        # Generate embeddings for chunks
        chunks_with_embeddings = await self.embedding_service.embed_chunks(chunks)
        
        # Generate embedding for full document
        document.embedding = await self.embedding_service.embed_text(document.content)
        
        # Store in database
        document_id = await self.vector_store.store_document(document)
        chunk_ids = await self.vector_store.store_chunks(chunks_with_embeddings)
        
        return document_id
    
    async def query_documents(
        self,
        query_request: QueryRequest
    ) -> QueryResponse:
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = await self.embedding_service.embed_query(query_request.query)
        
        # Perform similarity search
        results = await self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=query_request.top_k,
            metadata_filters=query_request.metadata_filters,
            similarity_threshold=query_request.similarity_threshold
        )
        
        # Format results
        formatted_results = []
        for chunk, similarity_score in results:
            result = {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "similarity_score": similarity_score,
                "metadata": chunk.chunk_metadata,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char
            }
            formatted_results.append(result)
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return QueryResponse(
            query=query_request.query,
            results=formatted_results,
            total_results=len(formatted_results),
            execution_time_ms=execution_time
        )
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        return await self.vector_store.get_document(document_id)
    
    async def delete_document(self, document_id: str) -> bool:
        return await self.vector_store.delete_document(document_id)
    
    async def get_documents_by_metadata(
        self,
        metadata_filters: Dict[str, Any],
        limit: int = 100
    ) -> List[Document]:
        return await self.vector_store.get_documents_by_metadata(
            metadata_filters=metadata_filters,
            limit=limit
        )
    
    async def close(self):
        await self.db_manager.close()