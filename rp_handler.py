#!/usr/bin/env python3
"""
RunPod Serverless Handler for RAG Pipeline

This handler provides a serverless interface for the RAG pipeline with the following endpoints:
- ingest: Ingest a document from URL
- query: Search documents with optional reranking
- health: Health check
- list_documents: List stored documents
"""

import runpod
import asyncio
import os
import traceback
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your RAG pipeline
from rag_pipeline.core.pipeline import RAGPipeline
from rag_pipeline.models.query import QueryRequest

# Global pipeline instance
pipeline: Optional[RAGPipeline] = None

async def initialize_pipeline():
    """Initialize the RAG pipeline with environment configuration."""
    global pipeline
    
    if pipeline is not None:
        return pipeline
    
    try:
        # Get configuration from environment variables
        database_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/rag_db")
        use_gpu = os.getenv("USE_GPU", "true").lower() == "true"
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        embedding_dimensions = os.getenv("EMBEDDING_DIMENSIONS")
        voyage_api_key = os.getenv("VOYAGE_API_KEY")
        use_reranker = os.getenv("USE_RERANKER", "false").lower() == "true"
        reranker_model = os.getenv("RERANKER_MODEL", "rerank-2.5")
        chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Convert embedding_dimensions to int if provided
        if embedding_dimensions:
            embedding_dimensions = int(embedding_dimensions)
        
        logger.info(f"Initializing pipeline with:")
        # logger.info(f"  - Database: {database_url}")
        logger.info(f"  - GPU: {use_gpu}")
        logger.info(f"  - Embedding Provider: {embedding_provider}")
        logger.info(f"  - Embedding Model: {embedding_model}")
        logger.info(f"  - Embedding Dimensions: {embedding_dimensions}")
        logger.info(f"  - Use Reranker: {use_reranker}")
        logger.info(f"  - Reranker Model: {reranker_model}")
        
        pipeline = RAGPipeline(
            database_url=database_url,
            use_gpu=use_gpu,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            voyage_api_key=voyage_api_key,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_reranker=use_reranker,
            reranker_model=reranker_model
        )
        
        await pipeline.initialize()
        logger.info("RAG Pipeline initialized successfully")
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        logger.error(traceback.format_exc())
        raise

async def handle_ingest(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle document ingestion request."""
    try:
        url = input_data.get("url")
        filename = input_data.get("filename")
        metadata = input_data.get("metadata", {})
        
        if not url or not filename:
            return {
                "error": "Missing required parameters: url and filename"
            }
        
        logger.info(f"Ingesting document: {filename} from {url}")
        
        pipeline = await initialize_pipeline()
        document_id = await pipeline.ingest_document(
            url=url,
            filename=filename,
            custom_metadata=metadata
        )
        
        return {
            "success": True,
            "document_id": document_id,
            "message": "Document ingested successfully"
        }
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "success": False
        }

async def handle_query(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle document query request."""
    try:
        query = input_data.get("query")
        top_k = input_data.get("top_k", 5)
        metadata_filters = input_data.get("metadata_filters", {})
        similarity_threshold = input_data.get("similarity_threshold", 0.0)
        
        if not query:
            return {
                "error": "Missing required parameter: query"
            }
        
        logger.info(f"Processing query: {query[:100]}...")
        
        pipeline = await initialize_pipeline()
        
        query_request = QueryRequest(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
            similarity_threshold=similarity_threshold
        )
        
        response = await pipeline.query_documents(query_request)
        
        # Convert response to dict
        return {
            "success": True,
            "query": response.query,
            "results": response.results,
            "total_results": response.total_results,
            "execution_time_ms": response.execution_time_ms
        }
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "success": False
        }

async def handle_list_documents(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle list documents request."""
    try:
        metadata_filters = input_data.get("metadata_filters", {})
        limit = input_data.get("limit", 100)
        
        logger.info(f"Listing documents with filters: {metadata_filters}")
        
        pipeline = await initialize_pipeline()
        documents = await pipeline.get_documents_by_metadata(
            metadata_filters=metadata_filters,
            limit=limit
        )
        
        # Convert documents to serializable format
        doc_list = []
        for doc in documents:
            doc_dict = {
                "id": doc.id,
                "metadata": {
                    "source_url": doc.metadata.source_url,
                    "filename": doc.metadata.filename,
                    "file_type": doc.metadata.file_type,
                    "file_size": doc.metadata.file_size,
                    "created_at": doc.metadata.created_at.isoformat() if doc.metadata.created_at else None,
                    "updated_at": doc.metadata.updated_at.isoformat() if doc.metadata.updated_at else None,
                    "custom_metadata": doc.metadata.custom_metadata
                }
            }
            doc_list.append(doc_dict)
        
        return {
            "success": True,
            "documents": doc_list,
            "total": len(doc_list)
        }
        
    except Exception as e:
        logger.error(f"List documents failed: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "success": False
        }

async def handle_health(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle health check request."""
    try:
        import torch
        
        # Check if pipeline is initialized
        pipeline_status = "initialized" if pipeline else "not_initialized"
        
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        # Check environment variables
        env_vars = {
            "DATABASE_URL": bool(os.getenv("DATABASE_URL")),
            "VOYAGE_API_KEY": bool(os.getenv("VOYAGE_API_KEY")),
            "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER", "sentence-transformers"),
            "USE_RERANKER": os.getenv("USE_RERANKER", "false")
        }
        
        return {
            "success": True,
            "status": "healthy",
            "pipeline_status": pipeline_status,
            "gpu_available": gpu_available,
            "gpu_count": gpu_count,
            "environment": env_vars
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e)
        }

def handler(event):
    """
    RunPod serverless handler function.
    
    Expected input format:
    {
        "action": "ingest|query|list_documents|health",
        "data": {
            # Action-specific parameters
        }
    }
    """
    try:
        logger.info(f"Received event: {event}")
        
        input_data = event.get("input", {})
        action = input_data.get("action")
        data = input_data.get("data", {})
        
        if not action:
            return {
                "error": "Missing required parameter: action",
                "success": False
            }
        
        # Run the appropriate async handler
        if action == "ingest":
            result = asyncio.run(handle_ingest(data))
        elif action == "query":
            result = asyncio.run(handle_query(data))
        elif action == "list_documents":
            result = asyncio.run(handle_list_documents(data))
        elif action == "health":
            result = asyncio.run(handle_health(data))
        else:
            result = {
                "error": f"Unknown action: {action}. Supported actions: ingest, query, list_documents, health",
                "success": False
            }
        
        logger.info(f"Handler result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "success": False
        }

if __name__ == "__main__":
    logger.info("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})