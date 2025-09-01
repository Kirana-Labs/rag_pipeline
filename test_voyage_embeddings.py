#!/usr/bin/env python3

import asyncio
import os
from rag_pipeline.core.embedder import EmbeddingService
from rag_pipeline.models.document import Document, DocumentMetadata, DocumentChunk


async def test_sentence_transformers():
    """Test the original sentence-transformers provider."""
    print("=== Testing Sentence Transformers ===")
    
    embedder = EmbeddingService(
        provider="sentence-transformers",
        model_name="all-MiniLM-L6-v2"
    )
    
    # Test single text embedding
    text = "This is a test sentence for embedding."
    embedding = await embedder.embed_text(text)
    print(f"Single text embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test multiple texts
    texts = [
        "First test sentence.",
        "Second test sentence about different content.",
        "Third sentence with more details."
    ]
    embeddings = await embedder.embed_texts(texts)
    print(f"Multiple texts embeddings: {len(embeddings)} vectors of {len(embeddings[0])} dimensions")
    
    await embedder.close()
    print("✓ Sentence transformers test completed\n")


async def test_voyage_embeddings():
    """Test the Voyage AI provider (requires API key)."""
    print("=== Testing Voyage AI Embeddings ===")
    
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("❌ VOYAGE_API_KEY not set, skipping Voyage tests")
        return
    
    try:
        embedder = EmbeddingService(
            provider="voyage",
            model_name="voyage-context-3",
            output_dimension=512,
            api_key=api_key
        )
        
        print(f"Expected embedding dimension: {embedder.get_embedding_dimension()}")
        
        # Test single text embedding
        text = "This is a test sentence for Voyage embedding."
        embedding = await embedder.embed_text(text)
        print(f"Single text embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        # Test query embedding
        query = "What is this document about?"
        query_embedding = await embedder.embed_query(query)
        print(f"Query embedding dimension: {len(query_embedding)}")
        
        # Test chunk embeddings (contextualized)
        chunks = [
            DocumentChunk(
                document_id="test-doc-1",
                content="This is the first chunk of a document about artificial intelligence.",
                chunk_index=0,
                start_char=0,
                end_char=60,
                chunk_metadata={"source": "test"}
            ),
            DocumentChunk(
                document_id="test-doc-1", 
                content="AI has many applications in healthcare, finance, and transportation.",
                chunk_index=1,
                start_char=61,
                end_char=125,
                chunk_metadata={"source": "test"}
            ),
            DocumentChunk(
                document_id="test-doc-2",
                content="Machine learning is a subset of artificial intelligence.",
                chunk_index=0,
                start_char=0,
                end_char=55,
                chunk_metadata={"source": "test"}
            )
        ]
        
        chunks_with_embeddings = await embedder.embed_chunks(chunks)
        print(f"Processed {len(chunks_with_embeddings)} chunks with contextualized embeddings")
        
        for i, chunk in enumerate(chunks_with_embeddings):
            if chunk.embedding:
                print(f"  Chunk {i+1}: {len(chunk.embedding)} dimensions, first 3 values: {chunk.embedding[:3]}")
            else:
                print(f"  Chunk {i+1}: No embedding generated!")
        
        await embedder.close()
        print("✓ Voyage AI test completed\n")
        
    except Exception as e:
        print(f"❌ Voyage AI test failed: {e}\n")


async def test_dimension_configurations():
    """Test different output dimensions for Voyage AI."""
    print("=== Testing Voyage AI Dimension Configurations ===")
    
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("❌ VOYAGE_API_KEY not set, skipping dimension tests")
        return
    
    dimensions = [256, 512, 1024]
    test_text = "Test text for dimension configuration."
    
    for dim in dimensions:
        try:
            embedder = EmbeddingService(
                provider="voyage",
                model_name="voyage-context-3",
                output_dimension=dim,
                api_key=api_key
            )
            
            embedding = await embedder.embed_text(test_text)
            print(f"Dimension {dim}: Got {len(embedding)} dimensions")
            
            await embedder.close()
            
        except Exception as e:
            print(f"❌ Dimension {dim} test failed: {e}")
    
    print("✓ Dimension configuration tests completed\n")


async def main():
    print("Testing Enhanced Embedding Service\n")
    
    # Test sentence transformers (should always work)
    await test_sentence_transformers()
    
    # Test Voyage AI (requires API key)
    await test_voyage_embeddings()
    
    # Test different dimensions
    await test_dimension_configurations()
    
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())