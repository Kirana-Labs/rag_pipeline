#!/usr/bin/env python3

import asyncio
import os
from rag_pipeline.core.reranker import RerankerService, RerankResult
from rag_pipeline.models.document import DocumentChunk


async def test_reranker():
    """Test the Voyage AI reranker functionality."""
    print("=== Testing Voyage AI Reranker ===")
    
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("❌ VOYAGE_API_KEY not set, skipping reranker tests")
        return
    
    try:
        reranker = RerankerService(
            provider="voyage",
            model="rerank-2.5",
            api_key=api_key
        )
        
        # Sample query
        query = "What are the applications of artificial intelligence?"
        
        # Sample document chunks with similarity scores
        chunks_with_scores = [
            (
                DocumentChunk(
                    document_id="doc-1",
                    content="Artificial intelligence has applications in healthcare, including medical diagnosis and drug discovery.",
                    chunk_index=0,
                    start_char=0,
                    end_char=100,
                    chunk_metadata={"source": "ai_overview.pdf"}
                ),
                0.85
            ),
            (
                DocumentChunk(
                    document_id="doc-1", 
                    content="Machine learning algorithms can analyze large datasets to identify patterns and make predictions.",
                    chunk_index=1,
                    start_char=101,
                    end_char=200,
                    chunk_metadata={"source": "ai_overview.pdf"}
                ),
                0.72
            ),
            (
                DocumentChunk(
                    document_id="doc-2",
                    content="The weather today is sunny with a temperature of 75 degrees Fahrenheit.",
                    chunk_index=0,
                    start_char=0,
                    end_char=70,
                    chunk_metadata={"source": "weather_report.txt"}
                ),
                0.20
            ),
            (
                DocumentChunk(
                    document_id="doc-1",
                    content="AI is used in autonomous vehicles for navigation and obstacle detection systems.",
                    chunk_index=2,
                    start_char=201,
                    end_char=280,
                    chunk_metadata={"source": "ai_overview.pdf"}
                ),
                0.78
            ),
            (
                DocumentChunk(
                    document_id="doc-3",
                    content="Financial institutions use AI for fraud detection and risk assessment.",
                    chunk_index=0,
                    start_char=0,
                    end_char=65,
                    chunk_metadata={"source": "fintech_ai.pdf"}
                ),
                0.80
            )
        ]
        
        print(f"Query: '{query}'")
        print(f"\nOriginal similarity-based ranking:")
        for i, (chunk, score) in enumerate(chunks_with_scores):
            print(f"  {i+1}. [Score: {score:.3f}] {chunk.content[:60]}...")
        
        # Test reranking
        print(f"\nTesting reranking...")
        reranked_results = await reranker.rerank(
            query=query,
            chunks_with_scores=chunks_with_scores,
            top_k=3
        )
        
        print(f"\nReranked results (top 3):")
        for result in reranked_results:
            print(f"  {result.rank}. [Relevance: {result.relevance_score:.3f}, Original: {result.original_similarity_score:.3f}]")
            print(f"     {result.chunk.content[:60]}...")
            print()
        
        # Test with different top_k
        print(f"Testing with all results (no top_k limit)...")
        all_reranked = await reranker.rerank(
            query=query,
            chunks_with_scores=chunks_with_scores
        )
        
        print(f"\nAll reranked results:")
        for result in all_reranked:
            improvement = "↑" if result.relevance_score > result.original_similarity_score else "↓"
            print(f"  {result.rank}. {improvement} [Rel: {result.relevance_score:.3f}, Orig: {result.original_similarity_score:.3f}]")
            print(f"     {result.chunk.content[:80]}...")
        
        # Test model info
        model_info = reranker.get_model_info()
        print(f"\nReranker model info: {model_info}")
        
        await reranker.close()
        print("\n✓ Reranker test completed successfully!")
        
    except Exception as e:
        print(f"❌ Reranker test failed: {e}")


async def test_empty_results():
    """Test reranker with empty input."""
    print("\n=== Testing Edge Cases ===")
    
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("❌ VOYAGE_API_KEY not set, skipping edge case tests")
        return
    
    try:
        reranker = RerankerService(provider="voyage", api_key=api_key)
        
        # Test empty results
        empty_results = await reranker.rerank(
            query="test query",
            chunks_with_scores=[],
            top_k=5
        )
        
        assert len(empty_results) == 0, "Empty input should return empty results"
        print("✓ Empty input test passed")
        
        await reranker.close()
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")


async def main():
    print("Testing Voyage AI Reranker Service\n")
    
    await test_reranker()
    await test_empty_results()
    
    print("\nAll reranker tests completed!")


if __name__ == "__main__":
    asyncio.run(main())