#!/usr/bin/env python3
"""
Simple test script to verify the RAG pipeline functionality.
Run this script to test the pipeline without starting the full API server.
"""

import asyncio
import os
from rag_pipeline.core.pipeline import RAGPipeline

async def test_pipeline():
    print("🚀 Testing RAG Pipeline...")
    
    # Use in-memory SQLite for testing (in production, use PostgreSQL)
    database_url = "sqlite+aiosqlite:///test_rag.db"
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        database_url=database_url,
        use_gpu=False,  # Disable GPU for testing
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=100
    )
    
    try:
        print("📚 Initializing pipeline...")
        await pipeline.initialize()
        
        # Test document ingestion
        print("📄 Testing document ingestion...")
        test_url = "https://www.python.org/dev/pep/pep-0008/"
        
        document_id = await pipeline.ingest_document(
            url=test_url,
            filename="pep8.html",
            custom_metadata={
                "category": "documentation",
                "topic": "python",
                "type": "coding_standards"
            }
        )
        
        print(f"✅ Document ingested successfully! ID: {document_id}")
        
        # Test querying
        print("🔍 Testing document query...")
        from rag_pipeline.models.query import QueryRequest
        
        query_request = QueryRequest(
            query="What are the naming conventions for functions?",
            top_k=3,
            similarity_threshold=0.5
        )
        
        response = await pipeline.query_documents(query_request)
        
        print(f"✅ Query successful! Found {response.total_results} results")
        for i, result in enumerate(response.results[:2], 1):
            print(f"  Result {i}: {result['content'][:100]}...")
            print(f"  Similarity: {result['similarity_score']:.3f}")
        
        # Test metadata filtering
        print("🏷️  Testing metadata filtering...")
        filtered_query = QueryRequest(
            query="python style guide",
            top_k=2,
            metadata_filters={"category": "documentation"},
            similarity_threshold=0.4
        )
        
        filtered_response = await pipeline.query_documents(filtered_query)
        print(f"✅ Filtered query successful! Found {filtered_response.total_results} results")
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await pipeline.close()
        
        # Clean up test database
        if os.path.exists("test_rag.db"):
            os.remove("test_rag.db")

if __name__ == "__main__":
    asyncio.run(test_pipeline())