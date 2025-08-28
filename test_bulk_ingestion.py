#!/usr/bin/env python3
"""
Test script for bulk ingestion functionality.
This script demonstrates how to use the bulk ingestion API endpoints.
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any


class BulkIngestionTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def test_bulk_ingestion(self):
        print("🚀 Testing Bulk Ingestion API...")
        
        # Test data - sample documents for bulk ingestion
        bulk_request = {
            "documents": [
                {
                    "url": "https://www.python.org/dev/pep/pep-0008/",
                    "filename": "pep8_style_guide.html",
                    "metadata": {
                        "category": "documentation",
                        "topic": "python",
                        "type": "style_guide",
                        "priority": "high"
                    }
                },
                {
                    "url": "https://docs.python.org/3/tutorial/introduction.html",
                    "filename": "python_intro.html", 
                    "metadata": {
                        "category": "documentation",
                        "topic": "python",
                        "type": "tutorial",
                        "priority": "medium"
                    }
                },
                {
                    "url": "https://docs.python.org/3/tutorial/classes.html",
                    "filename": "python_classes.html",
                    "metadata": {
                        "category": "documentation", 
                        "topic": "python",
                        "type": "tutorial",
                        "priority": "medium"
                    }
                }
            ],
            "batch_name": "python_docs_test_batch"
        }
        
        try:
            # Test 1: Submit bulk ingestion job
            print("\n📤 Test 1: Submitting bulk ingestion job...")
            response = await self.client.post(
                f"{self.base_url}/ingest/bulk",
                json=bulk_request,
                timeout=30.0
            )
            
            if response.status_code == 200:
                job_data = response.json()
                job_id = job_data["job_id"]
                print(f"✅ Bulk job created successfully!")
                print(f"   Job ID: {job_id}")
                print(f"   Status: {job_data['status']}")
                print(f"   Total documents: {job_data['total_documents']}")
                print(f"   Estimated time: {job_data.get('estimated_completion_time', 'Unknown')}")
            else:
                print(f"❌ Failed to create bulk job: {response.status_code}")
                print(f"   Response: {response.text}")
                return
            
            # Test 2: Monitor job progress
            print(f"\n📊 Test 2: Monitoring job progress...")
            max_checks = 20
            check_interval = 5  # seconds
            
            for i in range(max_checks):
                response = await self.client.get(f"{self.base_url}/ingest/bulk/{job_id}")
                
                if response.status_code == 200:
                    status_data = response.json()
                    status = status_data["status"]
                    processed = status_data["processed_documents"]
                    total = status_data["total_documents"]
                    successful = status_data["successful_documents"]
                    failed = status_data["failed_documents"]
                    
                    print(f"   Check {i+1}: {status} - {processed}/{total} processed "
                          f"({successful} success, {failed} failed)")
                    
                    if status in ["completed", "failed"]:
                        print(f"\n✅ Job {status}!")
                        
                        # Print detailed results
                        print("\n📋 Detailed Results:")
                        for j, result in enumerate(status_data["results"]):
                            print(f"   Document {j+1}: {result['filename']}")
                            print(f"     Status: {result['status']}")
                            if result.get('document_id'):
                                print(f"     Document ID: {result['document_id']}")
                            if result.get('error_message'):
                                print(f"     Error: {result['error_message']}")
                            if result.get('processing_time_seconds'):
                                print(f"     Processing time: {result['processing_time_seconds']:.2f}s")
                        
                        break
                else:
                    print(f"❌ Failed to get job status: {response.status_code}")
                    break
                
                if i < max_checks - 1:  # Don't wait after last check
                    await asyncio.sleep(check_interval)
            else:
                print(f"⏰ Job monitoring timed out after {max_checks * check_interval} seconds")
            
            # Test 3: List all bulk jobs
            print(f"\n📋 Test 3: Listing all bulk jobs...")
            response = await self.client.get(f"{self.base_url}/ingest/bulk?limit=10")
            
            if response.status_code == 200:
                jobs = response.json()
                print(f"✅ Found {len(jobs)} bulk jobs")
                for job in jobs[:3]:  # Show first 3 jobs
                    print(f"   Job: {job['job_id'][:8]}... | Status: {job['status']} | "
                          f"Docs: {job['processed_documents']}/{job['total_documents']}")
            else:
                print(f"❌ Failed to list jobs: {response.status_code}")
            
            # Test 4: Test querying ingested documents
            if job_data.get("status") == "completed":
                print(f"\n🔍 Test 4: Querying ingested documents...")
                
                query_request = {
                    "query": "What are Python naming conventions?",
                    "top_k": 3,
                    "metadata_filters": {
                        "category": "documentation",
                        "topic": "python"
                    }
                }
                
                response = await self.client.post(
                    f"{self.base_url}/query",
                    json=query_request
                )
                
                if response.status_code == 200:
                    query_data = response.json()
                    print(f"✅ Query successful! Found {query_data['total_results']} results")
                    print(f"   Execution time: {query_data['execution_time_ms']:.2f}ms")
                    
                    for i, result in enumerate(query_data["results"][:2], 1):
                        print(f"\n   Result {i}:")
                        print(f"     Similarity: {result['similarity_score']:.3f}")
                        print(f"     Content: {result['content'][:150]}...")
                        print(f"     Source: {result['metadata'].get('filename', 'Unknown')}")
                else:
                    print(f"❌ Query failed: {response.status_code}")
            
        except httpx.RequestError as e:
            print(f"❌ Request error: {e}")
            print("   Make sure the API server is running at http://localhost:8000")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    async def test_health_check(self):
        print("\n🏥 Testing health check...")
        try:
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API is healthy!")
                print(f"   Status: {health_data['status']}")
                print(f"   GPU Available: {health_data['gpu_available']}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
        return True
    
    async def run_tests(self):
        try:
            # First check if API is available
            if not await self.test_health_check():
                print("\n⚠️  API is not available. Please start the server first:")
                print("   python main.py")
                return
            
            await self.test_bulk_ingestion()
            print("\n🎉 Bulk ingestion tests completed!")
            
        finally:
            await self.client.aclose()


async def main():
    print("=" * 60)
    print("🧪 RAG Pipeline - Bulk Ingestion Test Suite")
    print("=" * 60)
    
    tester = BulkIngestionTester()
    await tester.run_tests()


if __name__ == "__main__":
    asyncio.run(main())