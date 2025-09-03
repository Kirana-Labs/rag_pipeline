import asyncio
import os
from rag_pipeline.core.pipeline import RAGPipeline

async def test_deduplication():
    # Use SQLite for testing (or PostgreSQL if DATABASE_URL is set)
    database_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///test_dedup.db")
    
    # Initialize the pipeline
    pipeline = RAGPipeline(
        database_url=database_url,
        use_gpu=False,
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=50
    )
    
    await pipeline.initialize()
    
    # Test URL (using a public document)
    test_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    print("Test 1: First ingestion of document")
    doc_id_1 = await pipeline.ingest_document(
        url=test_url,
        filename="test_document.pdf",
        custom_metadata={"project": "test", "version": "1.0"}
    )
    print(f"First ingestion - Document ID: {doc_id_1}")
    
    print("\nTest 2: Re-ingesting same document (default dedup by filename+filetype)")
    doc_id_2 = await pipeline.ingest_document(
        url=test_url,
        filename="test_document.pdf",
        custom_metadata={"project": "test", "version": "1.1"}
    )
    print(f"Second ingestion - Document ID: {doc_id_2}")
    print(f"Should be same as first: {doc_id_1 == doc_id_2}")
    
    print("\nTest 3: Ingesting with different filename (should create new document)")
    doc_id_3 = await pipeline.ingest_document(
        url=test_url,
        filename="test_document_v2.pdf",
        custom_metadata={"project": "test", "version": "2.0"}
    )
    print(f"Third ingestion - Document ID: {doc_id_3}")
    print(f"Should be different from first: {doc_id_1 != doc_id_3}")
    
    print("\nTest 4: Using custom dedup key (project field)")
    doc_id_4 = await pipeline.ingest_document(
        url=test_url,
        filename="another_name.pdf",
        custom_metadata={"project": "test", "version": "3.0"},
        dedup_key="project"
    )
    print(f"Fourth ingestion - Document ID: {doc_id_4}")
    print(f"Should match first (same project): {doc_id_4 == doc_id_1}")
    
    print("\nTest 5: Different custom dedup key value (should create new)")
    doc_id_5 = await pipeline.ingest_document(
        url=test_url,
        filename="yet_another.pdf",
        custom_metadata={"project": "production", "version": "1.0"},
        dedup_key="project"
    )
    print(f"Fifth ingestion - Document ID: {doc_id_5}")
    print(f"Should be different (different project): {doc_id_5 != doc_id_1}")
    
    await pipeline.close()
    
    print("\n✅ All deduplication tests completed!")

if __name__ == "__main__":
    asyncio.run(test_deduplication())