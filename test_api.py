import requests
import json
from collections import Counter

BASE_URL = "http://localhost:8000"  # Change to 8001 if using docker-compose

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.json()}")
    assert response.status_code == 200

def test_predict():
    """Test predict endpoint with Iris dataset features"""
    # Sample Iris features (sepal length, sepal width, petal length, petal width)
    test_data = {
        "features": [
            [5.1, 3.5, 1.4, 0.2],  # Expected: setosa (0)
            [6.7, 3.0, 5.2, 2.3],  # Expected: virginica (2)
            [5.9, 3.0, 4.2, 1.5]   # Expected: versicolor (1)
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    print(f"\nPredict endpoint:")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) == 3

def test_canary_deployment():
    """Test canary deployment by making multiple predictions"""
    test_data = {
        "features": [[5.1, 3.5, 1.4, 0.2]]
    }
    
    print("\n=== Testing Canary Deployment ===")
    
    # Set canary probability to 0.1 (10% canary/next, 90% stable/current)
    canary_config = {"probability": 0.1}
    response = requests.post(f"{BASE_URL}/set-canary-probability", json=canary_config)
    print(f"\nSet canary probability: {response.json()}")
    
    # Make 20 predictions and track which model is used
    model_usage = []
    for i in range(20):
        response = requests.post(f"{BASE_URL}/predict", json=test_data)
        model_used = response.json()["model_used"]
        model_usage.append(model_used)
    
    # Count usage
    usage_counts = Counter(model_usage)
    print(f"\nModel usage over 20 predictions (expect ~10% canary, ~90% stable):")
    for model, count in usage_counts.items():
        print(f"  {model}: {count} times ({count/20*100:.0f}%)")

def test_update_next_model():
    """Test updating the next model"""
    update_data = {"version": 1}
    
    response = requests.post(f"{BASE_URL}/update-model", json=update_data)
    print(f"\n=== Update next model ===")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200

def test_accept_next_model():
    """Test accepting next model as current"""
    response = requests.post(f"{BASE_URL}/accept-next-model")
    print(f"\n=== Accept next model as current ===")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200

if __name__ == "__main__":
    print("Testing ML Service API with Canary Deployment...\n")
    
    try:
        test_health()
        test_predict()
        test_canary_deployment()
        test_update_next_model()
        test_accept_next_model()
        
        # Test predict again after accepting
        print("\n=== Testing after accepting next model ===")
        test_predict()
        
        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to service. Make sure it's running on http://localhost:8001")
    except Exception as e:
        print(f"\n❌ Error: {e}")