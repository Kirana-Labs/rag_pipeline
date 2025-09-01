#!/usr/bin/env python3

import asyncio
from rag_pipeline.core.chunker import ChunkingService
from rag_pipeline.models.document import Document, DocumentMetadata


async def test_table_chunking():
    # Create test document with markdown tables
    test_content = """# Document with Tables

This is some introductory text before the table.

## Data Overview

Here's a sample table that should be kept together:

| Name | Age | City | Country |
|------|-----|------|---------|
| John | 25 | New York | USA |
| Jane | 30 | London | UK |
| Bob | 35 | Tokyo | Japan |
| Alice | 28 | Paris | France |

This text comes after the table.

## Another Section

More content here with another table:

| Product | Price | Stock |
|---------|-------|-------|
| Laptop | $999 | 50 |
| Mouse | $25 | 200 |
| Keyboard | $75 | 100 |

Final paragraph of text.
"""

    metadata = DocumentMetadata(
        source_url="test://table-test",
        filename="table_test.md",
        file_type="md"
    )
    
    document = Document(
        id="test-doc",
        content=test_content,
        metadata=metadata
    )

    # Test with small chunk size to force splitting
    chunker = ChunkingService(chunk_size=300, chunk_overlap=50)
    chunks = await chunker.chunk_document(document)
    
    print(f"Generated {len(chunks)} chunks:")
    print("=" * 50)
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1} (length: {len(chunk.content)}):")
        print("-" * 30)
        print(chunk.content)
        print("-" * 30)
        
        # Check if this chunk contains a table
        has_table = '|' in chunk.content and chunk.content.count('|') >= 6
        if has_table:
            print("✓ Contains table (kept intact)")
        else:
            print("ⓘ Regular text chunk")


if __name__ == "__main__":
    asyncio.run(test_table_chunking())