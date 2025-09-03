import time
from typing import Dict, Any, List, Optional, Tuple, Literal

from .document_processor import DocumentProcessor
from .chunker import ChunkingService
from .embedder import EmbeddingService
from .reranker import RerankerService
from ..storage.vector_store import VectorStore
from ..storage.database import DatabaseManager
from ..models.document import Document, DocumentChunk
from ..models.query import QueryRequest, QueryResponse


class RAGPipeline:
    def __init__(
        self,
        database_url: str,
        use_gpu: bool = True,
        embedding_provider: Literal["sentence-transformers", "voyage"] = "sentence-transformers",
        embedding_model: str = "all-MiniLM-L6-v2", 
        embedding_dimensions: Optional[int] = None,
        voyage_api_key: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_reranker: bool = False,
        reranker_model: str = "rerank-2.5"
    ):
        self.database_url = database_url
        self.use_gpu = use_gpu
        self.embedding_provider = embedding_provider
        self.use_reranker = use_reranker
        
        # Initialize components
        self.db_manager = DatabaseManager(database_url)
        self.vector_store = VectorStore(self.db_manager)
        self.document_processor = DocumentProcessor(use_gpu=use_gpu)
        self.chunking_service = ChunkingService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedding_service = EmbeddingService(
            provider=embedding_provider,
            model_name=embedding_model,
            device="cuda" if use_gpu else "mps",
            api_key=voyage_api_key,
            output_dimension=embedding_dimensions
        )
        
        # Initialize reranker if enabled
        self.reranker_service = None
        if use_reranker:
            self.reranker_service = RerankerService(
                provider="voyage",
                model=reranker_model,
                api_key=voyage_api_key
            )
    
    async def initialize(self):
        await self.db_manager.initialize_database()
    
    async def ingest_document(
        self,
        url: str,
        filename: str,
        custom_metadata: Optional[Dict[str, Any]] = None,
        dedup_key: Optional[str] = None
    ) -> str:
        # Determine file_type from filename for deduplication check
        file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
        
        # Check for existing document BEFORE downloading/processing
        existing_document_id = await self.vector_store.find_existing_document(
            dedup_key=dedup_key,
            filename=filename,
            file_type=file_extension,  # Use extension as initial file_type
            custom_metadata=custom_metadata
        )
        
        if existing_document_id:
            # Document already exists, return early with existing ID
            return existing_document_id
        
        # Process document only if it doesn't exist
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
        # document.embedding = await self.embedding_service.embed_text(document.content)
        
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
        
        # Perform similarity search - get more results if reranking is enabled
        search_top_k = query_request.top_k
        if self.use_reranker and self.reranker_service:
            # Get more results for reranking (typically 2-3x the final desired count)
            search_top_k = min(query_request.top_k * 3, 100)
        
        results = await self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=search_top_k,
            metadata_filters=query_request.metadata_filters,
            similarity_threshold=query_request.similarity_threshold
        )
        
        # Apply reranking if enabled
        if self.use_reranker and self.reranker_service and results:
            reranked_results = await self.reranker_service.rerank(
                query=query_request.query,
                chunks_with_scores=results,
                top_k=query_request.top_k
            )
            
            # Format reranked results
            formatted_results = []
            for rerank_result in reranked_results:
                result = {
                    "id": rerank_result.chunk.id,
                    "document_id": rerank_result.chunk.document_id,
                    "content": rerank_result.chunk.content,
                    "similarity_score": rerank_result.original_similarity_score,
                    "relevance_score": rerank_result.relevance_score,
                    "rank": rerank_result.rank,
                    "metadata": rerank_result.chunk.chunk_metadata,
                    "chunk_index": rerank_result.chunk.chunk_index,
                    "start_char": rerank_result.chunk.start_char,
                    "end_char": rerank_result.chunk.end_char,
                    "reranked": True
                }
                formatted_results.append(result)
        else:
            # Format results without reranking
            formatted_results = []
            for i, (chunk, similarity_score) in enumerate(results[:query_request.top_k]):
                result = {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "similarity_score": similarity_score,
                    "rank": i + 1,
                    "metadata": chunk.chunk_metadata,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "reranked": False
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
        await self.embedding_service.close()
        if self.reranker_service:
            await self.reranker_service.close()
        await self.db_manager.close()