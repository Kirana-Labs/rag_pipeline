import asyncio
import os
from typing import List, Optional, Dict, Any, Literal
import torch
from sentence_transformers import SentenceTransformer

try:
    import voyageai
except ImportError:
    voyageai = None

from ..models.document import DocumentChunk


class EmbeddingService:
    def __init__(
        self, 
        provider: Literal["sentence-transformers", "voyage"] = "sentence-transformers",
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        api_key: Optional[str] = None,
        output_dimension: Optional[int] = None
    ):
        self.provider = provider
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps")
        self.api_key = api_key
        self.output_dimension = output_dimension
        
        if provider == "sentence-transformers":
            self._load_sentence_transformer()
        elif provider == "voyage":
            self._setup_voyage()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _load_sentence_transformer(self):
        self.model = SentenceTransformer(self.model_name)
        self.model.to(self.device)
    
    def _setup_voyage(self):
        if voyageai is None:
            raise ImportError("voyageai package not installed. Install with: pip install voyageai")
            
        self.api_key = self.api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY environment variable or api_key parameter required for Voyage provider")
        
        if self.model_name == "all-MiniLM-L6-v2":
            self.model_name = "voyage-context-3"
        
        self.voyage_client = voyageai.AsyncClient(api_key=self.api_key)
        self._embedding_dimension = self.output_dimension or 512
    
    async def _voyage_embed_texts(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        """Embed texts using Voyage AI SDK."""
        #  or len(texts) == 1
        if input_type == "query":
            # Use regular embedding for queries or single texts
            result = await self.voyage_client.contextualized_embed(
                inputs=[texts],
                model=self.model_name,
                input_type=input_type,
                output_dimension=self.output_dimension
            )

            return [result.results[0].embeddings[0]]
        else:
            # Use contextualized embeddings for document chunks
            result = await self.voyage_client.contextualized_embed(
                inputs=[texts],  # inputs expects a list of lists
                model=self.model_name,
                input_type=input_type,
                output_dimension=self.output_dimension
            )
            
            return [emb for r in result.results for emb in r.embeddings]
    
    async def _voyage_embed_chunks_grouped(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Embed chunks using Voyage AI contextualized embeddings, grouping by document."""
        document_groups: Dict[str, List[DocumentChunk]] = {}
        
        for chunk in chunks:
            doc_id = chunk.document_id
            if doc_id not in document_groups:
                document_groups[doc_id] = []
            document_groups[doc_id].append(chunk)
        
        for doc_id, doc_chunks in document_groups.items():
            texts = [chunk.content for chunk in doc_chunks]
            embeddings = await self._voyage_embed_texts(texts, input_type="document")
            
            for chunk, embedding in zip(doc_chunks, embeddings):
                chunk.embedding = embedding
        
        return chunks

    async def embed_text(self, text: str) -> List[float]:
        if self.provider == "voyage":
            embeddings = await self._voyage_embed_texts([text], input_type="document")
            return embeddings[0]
        else:
            loop = asyncio.get_event_loop()
            
            def _embed():
                embedding = self.model.encode([text], convert_to_tensor=True)
                return embedding.cpu().numpy().tolist()[0]
            
            return await loop.run_in_executor(None, _embed)
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "voyage":
            return await self._voyage_embed_texts(texts, input_type="document")
        else:
            loop = asyncio.get_event_loop()
            
            def _embed():
                embeddings = self.model.encode(texts, convert_to_tensor=True)
                return embeddings.cpu().numpy().tolist()
            
            return await loop.run_in_executor(None, _embed)
    
    async def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        if self.provider == "voyage":
            return await self._voyage_embed_chunks_grouped(chunks)
        else:
            texts = [chunk.content for chunk in chunks]
            embeddings = await self.embed_texts(texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            return chunks
    
    async def embed_query(self, query: str) -> List[float]:
        if self.provider == "voyage":
            embeddings = await self._voyage_embed_texts([query], input_type="query")
            return embeddings[0]
        else:
            return await self.embed_text(query)
    
    def get_embedding_dimension(self) -> int:
        if self.provider == "voyage":
            return self._embedding_dimension
        else:
            return self.model.get_sentence_embedding_dimension()
    
    async def close(self):
        """Clean up resources."""
        if self.provider == "voyage" and hasattr(self, 'voyage_client'):
            # VoyageAI client doesn't require explicit cleanup in current version
            pass