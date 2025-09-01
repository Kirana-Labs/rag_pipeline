#!/usr/bin/env python3
"""
RunPod Client Helper

Easy-to-use client for testing the deployed RAG pipeline on RunPod serverless.
"""

import requests
import json
import time
from typing import Dict, Any, Optional, List

class RunPodClient:
    """Client for interacting with RunPod serverless RAG pipeline."""
    
    def __init__(self, endpoint_id: str, api_key: str):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def _make_request(self, endpoint: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        """Make a request to RunPod API."""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "success": False}
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the RAG pipeline."""
        payload = {
            "input": {
                "action": "health",
                "data": {}
            }
        }
        return self._make_request("runsync", payload)
    
    def ingest_document(
        self, 
        url: str, 
        filename: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ingest a document from URL."""
        payload = {
            "input": {
                "action": "ingest",
                "data": {
                    "url": url,
                    "filename": filename,
                    "metadata": metadata or {}
                }
            }
        }
        return self._make_request("runsync", payload, timeout=120)  # Longer timeout for ingestion
    
    def query_documents(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Query documents in the RAG pipeline."""
        payload = {
            "input": {
                "action": "query",
                "data": {
                    "query": query,
                    "top_k": top_k,
                    "metadata_filters": metadata_filters or {},
                    "similarity_threshold": similarity_threshold
                }
            }
        }
        return self._make_request("runsync", payload)
    
    def list_documents(
        self,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """List documents in the database."""
        payload = {
            "input": {
                "action": "list_documents",
                "data": {
                    "metadata_filters": metadata_filters or {},
                    "limit": limit
                }
            }
        }
        return self._make_request("runsync", payload)
    
    def run_async(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a request asynchronously and return job ID."""
        payload = {
            "input": {
                "action": action,
                "data": data
            }
        }
        return self._make_request("run", payload)
    
    def check_status(self, job_id: str) -> Dict[str, Any]:
        """Check the status of an async job."""
        return self._make_request(f"status/{job_id}", {})


def main():
    """Example usage of the RunPod client."""
    import os
    
    # Get credentials from environment
    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
    api_key = os.getenv("RUNPOD_API_KEY")
    
    if not endpoint_id or not api_key:
        print("Please set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY environment variables")
        print("\nExample:")
        print("export RUNPOD_ENDPOINT_ID=your-endpoint-id")
        print("export RUNPOD_API_KEY=your-api-key")
        return
    
    # Create client
    client = RunPodClient(endpoint_id, api_key)
    
    # Test health check
    print("=== Health Check ===")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    if not health.get("success"):
        print("❌ Health check failed. Please check your deployment.")
        return
    
    # List existing documents
    print("\n=== List Documents ===")
    docs = client.list_documents()
    print(f"Found {docs.get('total', 0)} documents")
    
    # Test ingestion (optional)
    test_ingestion = input("\nTest document ingestion? (y/N): ")
    if test_ingestion.lower().startswith('y'):
        print("\n=== Document Ingestion ===")
        result = client.ingest_document(
            url="https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            filename="test_document.pdf",
            metadata={"category": "test", "source": "w3.org"}
        )
        print(json.dumps(result, indent=2))
        
        if result.get("success"):
            print(f"✓ Document ingested with ID: {result.get('document_id')}")
        else:
            print(f"❌ Ingestion failed: {result.get('error')}")
    
    # Test query
    print("\n=== Query Documents ===")
    query_result = client.query_documents(
        query="What is this document about?",
        top_k=3
    )
    print(json.dumps(query_result, indent=2))
    
    if query_result.get("success"):
        results = query_result.get("results", [])
        print(f"\n✓ Found {len(results)} results:")
        for i, result in enumerate(results):
            score_info = f"Similarity: {result.get('similarity_score', 0):.3f}"
            if result.get("reranked"):
                score_info += f", Relevance: {result.get('relevance_score', 0):.3f}"
            print(f"  {i+1}. [{score_info}] {result.get('content', '')[:100]}...")
    
    print("\n✓ Testing completed!")


if __name__ == "__main__":
    main()