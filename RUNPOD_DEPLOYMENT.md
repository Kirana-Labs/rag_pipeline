# RunPod Serverless Deployment Guide

This guide explains how to deploy the RAG Pipeline to RunPod serverless with GPU support.

## Files Overview

- `Dockerfile` - Docker configuration for RunPod deployment
- `rp_handler.py` - RunPod serverless handler
- `.env.runpod` - Environment variables template

## Prerequisites

1. RunPod account with serverless access
2. Voyage AI API key (if using Voyage embeddings/reranking)
3. PostgreSQL database (can use RunPod's database services)
4. Docker installed locally (for testing)

## Deployment Steps

### 1. Prepare Environment Variables

Copy `.env.runpod` and update the values:

```bash
# Required
DATABASE_URL=postgresql://user:password@your-db-host:5432/rag_db
VOYAGE_API_KEY=your_voyage_api_key_here

# Optional (defaults provided)
USE_GPU=true
EMBEDDING_PROVIDER=voyage
EMBEDDING_MODEL=voyage-context-3
EMBEDDING_DIMENSIONS=512
USE_RERANKER=true
RERANKER_MODEL=rerank-2.5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### 2. Build and Test Docker Image Locally

```bash
# Build the Docker image
docker build -t rag-pipeline:latest .

# Test locally (optional)
docker run --env-file .env.runpod -p 8080:8080 rag-pipeline:latest
```

### 3. Deploy to RunPod

#### Option A: Using RunPod Web Interface

1. Go to RunPod Serverless dashboard
2. Click "New Endpoint"
3. Choose "Custom Image"
4. Enter your Docker image details
5. Set environment variables from `.env.runpod`
6. Configure GPU settings (recommend RTX 4090 or A100)
7. Set scaling settings (min 0, max as needed)
8. Deploy

#### Option B: Using RunPod CLI

```bash
# Install RunPod CLI
pip install runpod

# Deploy endpoint
runpod endpoint deploy \
  --name "rag-pipeline" \
  --image "your-registry/rag-pipeline:latest" \
  --gpu-type "NVIDIA RTX 4090" \
  --min-workers 0 \
  --max-workers 3 \
  --env DATABASE_URL=postgresql://... \
  --env VOYAGE_API_KEY=your_key
```

## API Usage

### Health Check

```python
import requests

response = requests.post('https://api.runpod.ai/v2/your-endpoint-id/runsync', json={
    "input": {
        "action": "health"
    }
})
print(response.json())
```

### Document Ingestion

```python
response = requests.post('https://api.runpod.ai/v2/your-endpoint-id/runsync', json={
    "input": {
        "action": "ingest",
        "data": {
            "url": "https://example.com/document.pdf",
            "filename": "document.pdf",
            "metadata": {
                "category": "research",
                "author": "John Doe"
            }
        }
    }
})
```

### Document Query

```python
response = requests.post('https://api.runpod.ai/v2/your-endpoint-id/runsync', json={
    "input": {
        "action": "query",
        "data": {
            "query": "What are the applications of AI?",
            "top_k": 5,
            "metadata_filters": {"category": "research"},
            "similarity_threshold": 0.7
        }
    }
})
```

### List Documents

```python
response = requests.post('https://api.runpod.ai/v2/your-endpoint-id/runsync', json={
    "input": {
        "action": "list_documents",
        "data": {
            "metadata_filters": {"category": "research"},
            "limit": 10
        }
    }
})
```

## Configuration Options

### Embedding Providers

**Sentence Transformers (Local)**
```env
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
USE_GPU=true
```

**Voyage AI (API-based)**
```env
EMBEDDING_PROVIDER=voyage
EMBEDDING_MODEL=voyage-context-3
EMBEDDING_DIMENSIONS=512
VOYAGE_API_KEY=your_key
```

### Reranking

Enable Voyage AI reranking for better search results:
```env
USE_RERANKER=true
RERANKER_MODEL=rerank-2.5
VOYAGE_API_KEY=your_key
```

## GPU Recommendations

- **RTX 4090**: Good performance, cost-effective
- **A100**: Best performance for heavy workloads
- **RTX 3090**: Budget option

## Scaling Configuration

- **Min Workers**: 0 (cost-effective, cold starts)
- **Max Workers**: 3-5 (adjust based on load)
- **GPU Memory**: 24GB recommended for large models

## Database Setup

You can use:
1. RunPod's PostgreSQL service
2. External managed database (AWS RDS, Google Cloud SQL)
3. Self-hosted PostgreSQL with pgvector extension

Ensure the database has the pgvector extension installed:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Monitoring and Logs

- Use RunPod's built-in monitoring dashboard
- Check logs for any initialization or runtime errors
- Monitor GPU utilization and memory usage

## Troubleshooting

### Common Issues

1. **Database connection failed**
   - Check DATABASE_URL format
   - Ensure database is accessible from RunPod
   - Verify pgvector extension is installed

2. **Voyage API errors**
   - Verify VOYAGE_API_KEY is correct
   - Check API quota and usage limits
   - Ensure API key has embeddings and reranking permissions

3. **GPU memory issues**
   - Reduce CHUNK_SIZE if processing large documents
   - Use smaller embedding models
   - Increase GPU memory allocation

4. **Cold start timeouts**
   - Consider keeping min_workers > 0 for critical applications
   - Optimize Docker image size
   - Pre-load models in the container

## Cost Optimization

- Use min_workers=0 for development/testing
- Choose appropriate GPU types based on workload
- Monitor usage patterns and adjust scaling settings
- Consider using lighter models (sentence-transformers) vs API calls (Voyage AI)

## Security Notes

- Store API keys securely in RunPod environment variables
- Use VPC or private networks for database connections
- Implement authentication for production use
- Regularly rotate API keys and database credentials