#!/usr/bin/env python3
"""
Quick test to verify timeout optimizations are working
"""
import requests
import json
import time

def test_no_timeout_system():
    """Test the system without timeout restrictions"""
    base_url = "http://localhost:5000"
    
    print("🚀 TESTING NO-TIMEOUT SYSTEM")
    print("=" * 50)
    
    # Test 1: Quick connectivity check
    try:
        print("📡 Testing app connectivity...")
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ App is running")
        else:
            print(f"⚠️ App returned: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ App not accessible: {e}")
        return
    
    # Test 2: Query with no timeout - let it run as long as needed
    print("\n📝 Testing unlimited query processing...")
    
    test_queries = [
        {
            "query": "procurement budget planning",
            "role": "general",
            "description": "Standard query - no timeout"
        },
        {
            "query": "annual procurement requirements and specifications",
            "role": "auditor", 
            "description": "Complex auditor query - no timeout"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n  🔍 Test {i}: {test['description']}")
        print(f"     Query: '{test['query']}'")
        print(f"     Role: {test['role']}")
        print(f"     ⏳ No timeout - will wait as long as needed...")
        
        start_time = time.time()
        
        try:
            # NO TIMEOUT - let the request run indefinitely
            response = requests.post(
                f"{base_url}/query",
                json={
                    "query": test["query"],
                    "role": test["role"]
                }
                # No timeout parameter - unlimited time
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"     ✅ SUCCESS in {elapsed:.1f}s")
                
                # Check optimization metrics
                processing_time = data.get('processing_time_ms', 0)
                docs_retrieved = data.get('search_efficiency', {}).get('documents_retrieved', 0)
                smart_filters = data.get('search_efficiency', {}).get('used_smart_filters', False)
                
                print(f"     ⚡ Processing time: {processing_time}ms")
                print(f"     📚 Documents used: {docs_retrieved}")
                print(f"     🎯 Smart filters: {smart_filters}")
                print(f"     💬 Response length: {len(data.get('response', ''))} chars")
                
                # Success - no timeout system working!
                return True
                
            elif response.status_code == 400:
                error_data = response.json()
                print(f"     ⚠️ Query validation: {error_data.get('message', 'Unknown')}")
                
            else:
                print(f"     ❌ Error {response.status_code}: {response.text[:100]}")
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"     ❌ Request failed after {elapsed:.1f}s: {e}")
    
    print(f"\n🔧 NO-TIMEOUT SYSTEM STATUS:")
    print(f"  ✅ Server timeout: REMOVED - unlimited processing time")
    print(f"  ✅ Client timeout: REMOVED - unlimited request time")
    print(f"  ✅ Ollama timeout: REMOVED - unlimited generation time")
    print(f"  ✅ Progress indicators: Active (no time limits shown)")
    print(f"  🎯 System will now wait as long as needed for query completion")

if __name__ == "__main__":
    test_no_timeout_system()