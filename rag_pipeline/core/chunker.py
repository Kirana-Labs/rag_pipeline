import re
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
        # if document.metadata.file_type.lower() in ['md', 'markdown']:
        return await self._chunk_markdown(document)
        # else:
        #     return await self._chunk_text(document)
    
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
                chunk_metadata={
                    "source_url": document.metadata.source_url,
                    "filename": document.metadata.filename,
                    "file_type": document.metadata.file_type,
                    **document.metadata.custom_metadata
                }
            )
            chunks.append(chunk)
            current_char = end_char - self.chunk_overlap
        
        return chunks
    
    def _extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract markdown tables from text and return their positions."""
        tables = []
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            if '|' in line and line.count('|') >= 2:
                table_start = i
                table_lines = [lines[i]]
                i += 1
                
                while i < len(lines) and '|' in lines[i].strip():
                    table_lines.append(lines[i])
                    i += 1
                
                if len(table_lines) >= 2:
                    table_content = '\n'.join(table_lines)
                    start_pos = sum(len(lines[j]) + 1 for j in range(table_start))
                    end_pos = start_pos + len(table_content)
                    
                    tables.append({
                        'start_line': table_start,
                        'end_line': i - 1,
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'content': table_content
                    })
            else:
                i += 1
        
        return tables
    
    def _split_text_preserving_tables(self, text: str) -> List[str]:
        """Split text while keeping tables intact."""
        tables = self._extract_tables(text)
        
        if not tables:
            return self.text_splitter.split_text(text)
        
        chunks = []
        current_pos = 0
        
        for table in tables:
            if current_pos < table['start_pos']:
                pre_table_text = text[current_pos:table['start_pos']].strip()
                if pre_table_text:
                    pre_table_chunks = self.text_splitter.split_text(pre_table_text)
                    chunks.extend(pre_table_chunks)
            
            if len(table['content']) <= self.chunk_size:
                chunks.append(table['content'])
            else:
                table_chunks = self.text_splitter.split_text(table['content'])
                chunks.extend(table_chunks)
            
            current_pos = table['end_pos']
        
        if current_pos < len(text):
            remaining_text = text[current_pos:].strip()
            if remaining_text:
                remaining_chunks = self.text_splitter.split_text(remaining_text)
                chunks.extend(remaining_chunks)
        
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
                sub_chunks = self._split_text_preserving_tables(chunk_text)
                for j, sub_chunk in enumerate(sub_chunks):
                    chunk = DocumentChunk(
                        document_id=document.id,
                        content=sub_chunk,
                        chunk_index=i * 100 + j,
                        start_char=start_char,
                        end_char=start_char + len(sub_chunk),
                        chunk_metadata=chunk_metadata
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
                    chunk_metadata=chunk_metadata
                )
                chunks.append(chunk)
            
            current_char = end_char
        
        return chunks