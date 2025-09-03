# RAG Pipeline with GPU-Accelerated Docling

A high-performance Retrieval-Augmented Generation (RAG) pipeline built with GPU-accelerated Docling and LangChain, designed for serverless environments. This pipeline can ingest various document formats (PDF, DOCX, XLSX, HTML, images) and provide fast semantic search with metadata filtering.

## Features

- 🚀 **GPU-accelerated document processing** with Docling
- 📄 **Multi-format support**: PDF, DOCX, XLSX, HTML, JPG, PNG
- 🔍 **Semantic search** with pgvector and sentence transformers
- 🏷️ **Metadata filtering** for precise document retrieval  
- 🔄 **Document deduplication** to prevent re-ingestion of identical documents
- ⚡ **Async API** built with FastAPI
- 🐳 **Docker support** with GPU acceleration
- 🗄️ **PostgreSQL + pgvector** for efficient vector storage
- 🧩 **Modular architecture** with clear separation of concerns

## Architecture

```
rag_pipeline/
├── core/                    # Core pipeline components
│   ├── document_processor.py  # GPU-accelerated document processing
│   ├── chunker.py             # Text chunking with LangChain
│   ├── embedder.py            # Sentence transformer embeddings
│   └── pipeline.py            # Main pipeline orchestrator
├── storage/                 # Database and vector storage
│   ├── database.py            # SQLAlchemy models and DB manager
│   └── vector_store.py        # pgvector operations
├── api/                     # FastAPI application
│   ├── app.py                 # Main API application
│   └── models.py              # Pydantic models
└── models/                  # Core data models
    ├── document.py            # Document and chunk models
    └── query.py               # Query request/response models
```

## Quick Start

### Prerequisites

- Python 3.13+
- PostgreSQL with pgvector extension
- NVIDIA GPU (optional, but recommended)
- CUDA toolkit (if using GPU)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd 2025-08-28-g2x-rag-pipeline

# Install dependencies
pip install uv
uv pip install -e .
```

### 2. Database Setup

Start PostgreSQL with pgvector:

```bash
docker-compose up postgres -d
```

Or manually create database:

```sql
CREATE DATABASE rag_db;
CREATE EXTENSION vector;
```

### 3. Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/rag_db
USE_GPU=true
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 4. Run the API

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Ingest Single Document

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/document.pdf",
    "filename": "example.pdf",
    "metadata": {
      "category": "research",
      "author": "John Doe",
      "tags": ["AI", "ML"]
    }
  }'
```

#### Document Deduplication

By default, documents are deduplicated by filename + file type. You can specify a custom metadata field for deduplication:

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/document.pdf",
    "filename": "example.pdf",
    "metadata": {
      "document_id": "doc_12345",
      "category": "research"
    },
    "dedup_key": "document_id"
  }'
```

When a duplicate is detected, the API returns the existing document ID without re-downloading or re-processing the file.

### Bulk Ingest Documents

Submit multiple documents for processing in the background:

```bash
curl -X POST "http://localhost:8000/ingest/bulk" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "url": "https://example.com/doc1.pdf",
        "filename": "document1.pdf",
        "metadata": {"category": "research", "priority": "high"},
        "dedup_key": null  // Uses default filename+filetype dedup
      },
      {
        "url": "https://example.com/doc2.pdf", 
        "filename": "document2.pdf",
        "metadata": {"category": "research", "priority": "medium", "doc_id": "unique_123"},
        "dedup_key": "doc_id"  // Uses custom field for dedup
      }
    ],
    "batch_name": "research_papers_batch_1"
  }'
```

Bulk ingestion also supports per-document deduplication settings.

### Check Bulk Job Status

```bash
# Get specific job status
curl "http://localhost:8000/ingest/bulk/{job_id}"

# List all bulk jobs
curl "http://localhost:8000/ingest/bulk?limit=10"

# Cancel a running job
curl -X DELETE "http://localhost:8000/ingest/bulk/{job_id}"
```

### Query Documents

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 5,
    "metadata_filters": {
      "category": "research"
    },
    "similarity_threshold": 0.7
  }'
```

### List Documents

```bash
curl "http://localhost:8000/documents?category=research&limit=10"
```

### Get Specific Document

```bash
curl "http://localhost:8000/documents/{document_id}"
```

### Delete Document

```bash
curl -X DELETE "http://localhost:8000/documents/{document_id}"
```

## Docker Deployment

### Build and Run

```bash
docker-compose up --build
```

### GPU Support

For GPU acceleration, uncomment the GPU configuration in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DATABASE_URL` | postgresql://user:password@localhost:5432/rag_db | PostgreSQL connection string |
| `USE_GPU` | true | Enable GPU acceleration |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `CHUNK_SIZE` | 1000 | Text chunk size |
| `CHUNK_OVERLAP` | 200 | Chunk overlap size |
| `MAX_CONCURRENT_BULK_JOBS` | 3 | Maximum concurrent bulk ingestion jobs |
| `MAX_CONCURRENT_DOCS_PER_JOB` | 5 | Maximum concurrent documents per bulk job |
| `HOST` | 0.0.0.0 | API host |
| `PORT` | 8000 | API port |

## Supported File Types

- **PDF**: Full text extraction with OCR support
- **DOCX**: Microsoft Word documents
- **XLSX**: Excel spreadsheets (text content)
- **HTML**: Web pages and HTML files
- **Images**: JPG, PNG with OCR text extraction

## Performance Features

### GPU Acceleration
- Document processing with CUDA-enabled Docling
- GPU-accelerated embeddings with sentence transformers
- Optimized for serverless GPU environments

### Efficient Storage
- pgvector for fast similarity search
- Indexed metadata for filtering
- Chunked storage for large documents

### Async Processing
- Non-blocking document ingestion
- Concurrent request handling
- Background processing support

### Bulk Ingestion
- Process multiple documents in background jobs
- Configurable concurrency limits
- Real-time job progress monitoring
- Batch processing with metadata support
- Job cancellation and cleanup capabilities

### Document Deduplication
- Prevents re-ingestion of identical documents
- Checks for duplicates before downloading/processing files
- Flexible deduplication strategies:
  - Default: filename + file type combination
  - Custom: any metadata field (e.g., document_id, url, hash)
- Returns existing document ID for duplicates
- Saves bandwidth and processing time for large files

## Development

### Project Structure

The codebase follows a modular architecture:

- **Core Components**: Document processing, chunking, and embedding
- **Storage Layer**: Database management and vector operations
- **API Layer**: FastAPI application with async endpoints
- **Data Models**: Pydantic models for type safety

### Adding New Document Types

1. Extend `DocumentProcessor.process_file()` method
2. Add file type detection logic
3. Update supported formats documentation

### Customizing Chunking

Modify `ChunkingService` parameters:

```python
chunking_service = ChunkingService(
    chunk_size=1500,
    chunk_overlap=300,
    separators=["\n\n", "\n", ".", " "]
)
```

### Custom Embeddings

Replace the embedding model:

```python
embedding_service = EmbeddingService(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
```

## API Documentation

Once running, visit:
- Interactive docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or disable GPU
2. **Database connection errors**: Check PostgreSQL status and credentials
3. **Slow processing**: Enable GPU acceleration and check CUDA installation

### Logs

Check application logs for detailed error information:

```bash
docker-compose logs rag-api
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.