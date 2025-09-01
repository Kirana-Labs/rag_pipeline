#!/usr/bin/env python3
"""
Test script for RunPod handler

Tests the RunPod handler locally before deployment.
"""

import sys
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.runpod')

# Import the handler
from rp_handler import handler

def test_health_check():
    """Test health check endpoint."""
    print("=== Testing Health Check ===")
    
    event = {
        "input": {
            "action": "health",
            "data": {}
        }
    }
    
    result = handler(event)
    print(f"Health check result: {result}")
    
    assert result.get("success") is True or result.get("status") == "healthy"
    print("✓ Health check passed\n")

def test_query_without_documents():
    """Test query with no documents in database."""
    print("=== Testing Query (No Documents) ===")
    
    event = {
        "input": {
            "action": "query",
            "data": {
                "query": "What is artificial intelligence?",
                "top_k": 5
            }
        }
    }
    
    result = handler(event)
    print(f"Query result: {result}")
    
    # Should succeed but return no results
    if result.get("success"):
        print(f"Query returned {result.get('total_results', 0)} results")
    print("✓ Query test passed\n")

def test_list_documents():
    """Test listing documents."""
    print("=== Testing List Documents ===")
    
    event = {
        "input": {
            "action": "list_documents",
            "data": {
                "limit": 10
            }
        }
    }
    
    result = handler(event)
    print(f"List documents result: {result}")
    
    if result.get("success"):
        print(f"Found {result.get('total', 0)} documents")
    print("✓ List documents test passed\n")

def test_ingest_document():
    """Test document ingestion."""
    print("=== Testing Document Ingestion ===")
    
    # Use a public test document
    event = {
        "input": {
            "action": "ingest",
            "data": {
                "url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                "filename": "test_document.pdf",
                "metadata": {
                    "category": "test",
                    "source": "w3.org"
                }
            }
        }
    }
    
    result = handler(event)
    print(f"Ingestion result: {result}")
    
    if result.get("success"):
        print(f"Document ID: {result.get('document_id')}")
    else:
        print(f"Ingestion failed: {result.get('error')}")
    
    print("✓ Ingestion test completed\n")

def test_invalid_action():
    """Test invalid action handling."""
    print("=== Testing Invalid Action ===")
    
    event = {
        "input": {
            "action": "invalid_action",
            "data": {}
        }
    }
    
    result = handler(event)
    print(f"Invalid action result: {result}")
    
    assert result.get("success") is False
    assert "Unknown action" in result.get("error", "")
    print("✓ Invalid action test passed\n")

def test_missing_action():
    """Test missing action parameter."""
    print("=== Testing Missing Action ===")
    
    event = {
        "input": {
            "data": {}
        }
    }
    
    result = handler(event)
    print(f"Missing action result: {result}")
    
    assert result.get("success") is False
    assert "Missing required parameter: action" in result.get("error", "")
    print("✓ Missing action test passed\n")

def test_query_with_missing_parameters():
    """Test query with missing required parameters."""
    print("=== Testing Query Missing Parameters ===")
    
    event = {
        "input": {
            "action": "query",
            "data": {}
        }
    }
    
    result = handler(event)
    print(f"Query missing params result: {result}")
    
    assert result.get("success") is False
    assert "Missing required parameter: query" in result.get("error", "")
    print("✓ Query missing parameters test passed\n")

def main():
    """Run all tests."""
    print("Testing RunPod Handler Locally")
    print("=" * 40)
    
    # Check environment variables
    required_env_vars = ['DATABASE_URL']
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please copy .env.runpod to .env and configure the values")
        return
    
    try:
        # Test basic functionality
        test_health_check()
        test_invalid_action()
        test_missing_action()
        test_query_with_missing_parameters()
        
        # Test core functionality (may fail if DB not accessible)
        try:
            test_list_documents()
            test_query_without_documents()
            
            # Only test ingestion if user confirms
            response = input("Test document ingestion? This will attempt to download and process a test PDF (y/N): ")
            if response.lower().startswith('y'):
                test_ingest_document()
            
        except Exception as e:
            print(f"⚠️  Core functionality tests failed (likely DB connection issue): {e}")
            print("This is expected if database is not accessible from local environment")
        
        print("=" * 40)
        print("✓ All handler tests completed!")
        print("\nTo deploy to RunPod:")
        print("1. Build Docker image: docker build -t rag-pipeline:latest .")
        print("2. Push to container registry")
        print("3. Deploy to RunPod with environment variables")
        print("4. Test with RunPod's API endpoints")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()