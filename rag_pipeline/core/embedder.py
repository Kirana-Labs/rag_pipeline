import asyncio
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer

from ..models.document import DocumentChunk


class EmbeddingService:
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps")
        self._load_model()
    
    def _load_model(self):
        self.model = SentenceTransformer(self.model_name)
        self.model.to(self.device)
    
    async def embed_text(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        
        def _embed():
            embedding = self.model.encode([text], convert_to_tensor=True)
            return embedding.cpu().numpy().tolist()[0]
        
        return await loop.run_in_executor(None, _embed)
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        
        def _embed():
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            return embeddings.cpu().numpy().tolist()
        
        return await loop.run_in_executor(None, _embed)
    
    async def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embed_texts(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
    
    async def embed_query(self, query: str) -> List[float]:
        return await self.embed_text(query)
    
    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()