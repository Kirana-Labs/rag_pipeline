#!/usr/bin/env python3

import asyncio
from rag_pipeline.core.chunker import ChunkingService
from rag_pipeline.models.document import Document, DocumentMetadata


async def test_large_table_chunking():
    # Create test document with a large table that exceeds chunk size
    large_table = """# Large Table Test

This is some text before a very large table.

| ID | Name | Email | Department | Role | Salary | Start Date | Manager | Location | Phone |
|----|------|-------|------------|------|--------|------------|---------|----------|-------|
| 1 | John Smith | john.smith@company.com | Engineering | Senior Developer | $95000 | 2020-01-15 | Jane Doe | New York | 555-0101 |
| 2 | Jane Doe | jane.doe@company.com | Engineering | Team Lead | $110000 | 2019-03-10 | Bob Johnson | New York | 555-0102 |
| 3 | Bob Johnson | bob.johnson@company.com | Engineering | Manager | $125000 | 2018-06-01 | Alice Wilson | New York | 555-0103 |
| 4 | Alice Wilson | alice.wilson@company.com | Engineering | Director | $150000 | 2017-09-12 | CEO | New York | 555-0104 |
| 5 | Charlie Brown | charlie.brown@company.com | Marketing | Specialist | $65000 | 2021-11-20 | David Lee | Los Angeles | 555-0105 |
| 6 | David Lee | david.lee@company.com | Marketing | Manager | $95000 | 2020-05-14 | Alice Wilson | Los Angeles | 555-0106 |
| 7 | Emma Davis | emma.davis@company.com | Sales | Representative | $55000 | 2022-02-28 | Frank Miller | Chicago | 555-0107 |
| 8 | Frank Miller | frank.miller@company.com | Sales | Manager | $85000 | 2019-12-03 | Alice Wilson | Chicago | 555-0108 |

This is text after the large table.
"""

    metadata = DocumentMetadata(
        source_url="test://large-table-test",
        filename="large_table_test.md", 
        file_type="md"
    )
    
    document = Document(
        id="test-large-doc",
        content=large_table,
        metadata=metadata
    )

    # Test with very small chunk size to see table splitting behavior
    chunker = ChunkingService(chunk_size=200, chunk_overlap=50)
    chunks = await chunker.chunk_document(document)
    
    print(f"Generated {len(chunks)} chunks with small chunk size (200):")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1} (length: {len(chunk.content)}):")
        print("-" * 40)
        print(chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content)
        print("-" * 40)
        
        # Check if this chunk contains table parts
        has_table_header = '|' in chunk.content and '---' in chunk.content
        has_table_rows = '|' in chunk.content and chunk.content.count('|') >= 6
        
        if has_table_header:
            print("✓ Contains table header")
        elif has_table_rows:
            print("✓ Contains table rows")
        else:
            print("ⓘ Regular text chunk")


if __name__ == "__main__":
    asyncio.run(test_large_table_chunking())