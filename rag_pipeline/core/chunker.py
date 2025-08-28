from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

from ..models.document import Document, DocumentChunk


class ChunkingService:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        
        self._setup_splitters()
    
    def _setup_splitters(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
        
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
    
    async def chunk_document(self, document: Document) -> List[DocumentChunk]:
        if document.metadata.file_type.lower() in ['md', 'markdown']:
            return await self._chunk_markdown(document)
        else:
            return await self._chunk_text(document)
    
    async def _chunk_text(self, document: Document) -> List[DocumentChunk]:
        text_chunks = self.text_splitter.split_text(document.content)
        
        chunks = []
        current_char = 0
        
        for i, chunk_text in enumerate(text_chunks):
            start_char = current_char
            end_char = current_char + len(chunk_text)
            
            chunk = DocumentChunk(
                document_id=document.id,
                content=chunk_text,
                chunk_index=i,
                start_char=start_char,
                end_char=end_char,
                metadata={
                    "source_url": document.metadata.source_url,
                    "filename": document.metadata.filename,
                    "file_type": document.metadata.file_type,
                    **document.metadata.custom_metadata
                }
            )
            chunks.append(chunk)
            current_char = end_char - self.chunk_overlap
        
        return chunks
    
    async def _chunk_markdown(self, document: Document) -> List[DocumentChunk]:
        md_header_splits = self.markdown_splitter.split_text(document.content)
        
        chunks = []
        current_char = 0
        
        for i, doc in enumerate(md_header_splits):
            chunk_text = doc.page_content
            metadata = doc.metadata
            
            start_char = current_char
            end_char = current_char + len(chunk_text)
            
            chunk_metadata = {
                "source_url": document.metadata.source_url,
                "filename": document.metadata.filename,
                "file_type": document.metadata.file_type,
                **document.metadata.custom_metadata,
                **metadata
            }
            
            if len(chunk_text) > self.chunk_size:
                sub_chunks = self.text_splitter.split_text(chunk_text)
                for j, sub_chunk in enumerate(sub_chunks):
                    chunk = DocumentChunk(
                        document_id=document.id,
                        content=sub_chunk,
                        chunk_index=i * 100 + j,
                        start_char=start_char,
                        end_char=start_char + len(sub_chunk),
                        metadata=chunk_metadata
                    )
                    chunks.append(chunk)
                    start_char += len(sub_chunk) - self.chunk_overlap
            else:
                chunk = DocumentChunk(
                    document_id=document.id,
                    content=chunk_text,
                    chunk_index=i,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)
            
            current_char = end_char
        
        return chunks