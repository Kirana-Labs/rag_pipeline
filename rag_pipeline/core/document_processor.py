import asyncio
import httpx
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile
import os
from PIL import Image

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions, EasyOcrOptions
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend

from ..models.document import Document, DocumentMetadata


class DocumentProcessor:
    def __init__(self, use_gpu: bool = True, model_device: str = "cuda"):
        self.use_gpu = use_gpu
        self.model_device = model_device
        self._setup_converter()
    
    def _setup_converter(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        # ocr_options = TesseractOcrOptions(force_full_page_ocr=True)
        # Uncomment this to force full page OCR, which is much slower but more accurate
        # for PDFs with images.
        # ocr_options = EasyOcrOptions(force_full_page_ocr=True)
        # pipeline_options.ocr_options = ocr_options
        
        if self.use_gpu:
            pipeline_options.accelerator_options.device = self.model_device
        
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
            }
        )
    
    async def process_document_from_url(
        self, 
        url: str, 
        filename: str, 
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            file_size = len(response.content)
            file_type = self._determine_file_type(filename, content_type)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            try:
                content = await self._process_file(temp_file_path, file_type)
                
                metadata = DocumentMetadata(
                    source_url=url,
                    filename=filename,
                    file_type=file_type,
                    file_size=file_size,
                    custom_metadata=custom_metadata or {}
                )
                
                return Document(content=content, metadata=metadata)
            
            finally:
                os.unlink(temp_file_path)
    
    async def _process_file(self, file_path: str, file_type: str) -> str:
        return await self._process_with_docling(file_path)
    
    async def _process_with_docling(self, file_path: str) -> str:
        loop = asyncio.get_event_loop()
        
        def _convert():
            result = self.doc_converter.convert(file_path)
            return result.document.export_to_markdown()
        
        return await loop.run_in_executor(None, _convert)
    
    async def _process_image(self, file_path: str) -> str:
        loop = asyncio.get_event_loop()
        
        def _extract_text():
            try:
                from PIL import Image
                import pytesseract
                
                with Image.open(file_path) as img:
                    text = pytesseract.image_to_string(img)
                    return f"Image content:\n{text}"
            except ImportError:
                return f"Image file: {Path(file_path).name} (OCR not available - install pytesseract)"
        
        return await loop.run_in_executor(None, _extract_text)
    
    def _determine_file_type(self, filename: str, content_type: str) -> str:
        if content_type:
            type_mapping = {
                'application/pdf': 'pdf',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                'text/html': 'html',
                'image/jpeg': 'jpg',
                'image/png': 'png'
            }
            if content_type in type_mapping:
                return type_mapping[content_type]
        
        extension = Path(filename).suffix.lower().lstrip('.')
        return extension if extension else 'unknown'