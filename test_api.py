#!/usr/bin/env python3
"""
Test script for Demand Forecast API
Can test both local and Cloud Run deployments
"""

import requests
import json
import sys
from datetime import datetime

def test_api(base_url):
    """Test all API endpoints"""
    print(f"=== Testing Demand Forecast API ===")
    print(f"Base URL: {base_url}\n")
    
    # Test 1: Root endpoint
    print("1. Testing root endpoint...")
    try:
        response = requests.get(base_url)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Health check
    print("\n2. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Model info
    print("\n3. Testing model info...")
    try:
        response = requests.get(f"{base_url}/model_info")
        print(f"   Status: {response.status_code}")
        data = response.json()
        if data.get('success'):
            print(f"   Best Model: {data['model_info']['best_model']}")
            print(f"   Available Models: {data['model_info']['available_models']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 4: Predictions
    print("\n4. Testing predictions...")
    try:
        payload = {
            "days_ahead": 7,
            "include_bounds": True
        }
        response = requests.post(
            f"{base_url}/predict",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        print(f"   Status: {response.status_code}")
        data = response.json()
        if data.get('success'):
            print(f"   Forecast days: {len(data['forecast'])}")
            print(f"   Total shipments: {data['summary']['total_shipments']:,}")
            print(f"   Daily average: {data['summary']['daily_average']:.1f}")
            print(f"   Peak day: {data['summary']['peak_day']} ({data['summary']['peak_value']:,})")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 5: SKU predictions
    print("\n5. Testing SKU predictions...")
    try:
        payload = {
            "top_n": 5,
            "days_ahead": 30
        }
        response = requests.post(
            f"{base_url}/predict_sku",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        print(f"   Status: {response.status_code}")
        data = response.json()
        if data.get('success'):
            print(f"   SKUs analyzed: {data['summary']['skus_analyzed']}")
            print(f"   High growth SKUs: {data['summary']['high_growth_skus']}")
            print("\n   Top 3 SKUs:")
            for sku in data['sku_forecast'][:3]:
                print(f"   - SKU {sku['sku_id']}: {sku['predicted_total']:,} units "
                      f"(Growth: {sku['growth_rate']}%)")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n=== API Testing Complete ===")


def performance_test(base_url, num_requests=10):
    """Test API performance"""
    print(f"\n=== Performance Test ({num_requests} requests) ===")
    
    import time
    times = []
    
    for i in range(num_requests):
        start = time.time()
        response = requests.post(
            f"{base_url}/predict",
            json={"days_ahead": 7},
            headers={'Content-Type': 'application/json'}
        )
        end = time.time()
        
        if response.status_code == 200:
            times.append(end - start)
            print(f"Request {i+1}: {(end-start)*1000:.0f}ms")
        else:
            print(f"Request {i+1}: Failed (Status {response.status_code})")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\nAverage response time: {avg_time*1000:.0f}ms")
        print(f"Min: {min(times)*1000:.0f}ms")
        print(f"Max: {max(times)*1000:.0f}ms")


if __name__ == "__main__":
    # Default to local testing
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8080"
    
    # Run tests
    test_api(base_url)
    
    # Optional: Run performance test
    if len(sys.argv) > 2 and sys.argv[2] == "--perf":
        performance_test(base_url)