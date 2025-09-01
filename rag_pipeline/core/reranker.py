import os
from typing import List, Optional, Tuple, Literal
from dataclasses import dataclass

try:
    import voyageai
except ImportError:
    voyageai = None

from ..models.document import DocumentChunk


@dataclass
class RerankResult:
    """Result from reranking operation."""
    chunk: DocumentChunk
    relevance_score: float
    original_similarity_score: float
    rank: int


class RerankerService:
    """Service for reranking retrieved documents using Voyage AI."""
    
    def __init__(
        self,
        provider: Literal["voyage"] = "voyage",
        model: str = "rerank-2.5",
        api_key: Optional[str] = None,
        max_documents: int = 100
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.max_documents = max_documents
        
        if provider == "voyage":
            self._setup_voyage()
        else:
            raise ValueError(f"Unsupported reranker provider: {provider}")
    
    def _setup_voyage(self):
        """Setup Voyage AI reranker client."""
        if voyageai is None:
            raise ImportError("voyageai package not installed. Install with: pip install voyageai")
        
        self.api_key = self.api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY environment variable or api_key parameter required")
        
        self.voyage_client = voyageai.AsyncClient(api_key=self.api_key)
    
    async def rerank(
        self,
        query: str,
        chunks_with_scores: List[Tuple[DocumentChunk, float]],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank retrieved chunks based on their relevance to the query.
        
        Args:
            query: The search query
            chunks_with_scores: List of (chunk, similarity_score) tuples from initial retrieval
            top_k: Number of top results to return (defaults to all)
        
        Returns:
            List of RerankResult objects sorted by relevance
        """
        if not chunks_with_scores:
            return []
        
        # Limit documents to max_documents for API constraints
        if len(chunks_with_scores) > self.max_documents:
            chunks_with_scores = chunks_with_scores[:self.max_documents]
        
        # Extract documents for reranking
        documents = [chunk.content for chunk, _ in chunks_with_scores]
        
        if self.provider == "voyage":
            return await self._voyage_rerank(query, chunks_with_scores, documents, top_k)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _voyage_rerank(
        self,
        query: str,
        chunks_with_scores: List[Tuple[DocumentChunk, float]],
        documents: List[str],
        top_k: Optional[int]
    ) -> List[RerankResult]:
        """Rerank using Voyage AI reranker."""
        try:
            # Call Voyage AI reranker
            rerank_result = await self.voyage_client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_k=top_k,
                truncation=True  # Automatically handle long documents
            )
            
            # Build results with reranking scores
            results = []
            for result_item in rerank_result.results:
                index = result_item.index
                relevance_score = result_item.relevance_score
                
                # Get original chunk and similarity score
                original_chunk, original_score = chunks_with_scores[index]
                
                rerank_result_obj = RerankResult(
                    chunk=original_chunk,
                    relevance_score=relevance_score,
                    original_similarity_score=original_score,
                    rank=len(results) + 1
                )
                results.append(rerank_result_obj)
            
            return results
            
        except Exception as e:
            # Fallback to original ordering if reranking fails
            print(f"Reranking failed, falling back to similarity scores: {e}")
            return [
                RerankResult(
                    chunk=chunk,
                    relevance_score=similarity_score,  # Use similarity as relevance
                    original_similarity_score=similarity_score,
                    rank=i + 1
                )
                for i, (chunk, similarity_score) in enumerate(chunks_with_scores[:top_k] if top_k else chunks_with_scores)
            ]
    
    async def close(self):
        """Clean up resources."""
        if self.provider == "voyage":
            # VoyageAI client doesn't require explicit cleanup
            pass
    
    def get_model_info(self) -> dict:
        """Get information about the current reranker model."""
        return {
            "provider": self.provider,
            "model": self.model,
            "max_documents": self.max_documents
        }