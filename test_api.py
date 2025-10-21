import requests
import json

BASE_URL = "http://localhost:8000"

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

def test_update_model():
    """Test update-model endpoint"""
    update_data = {
        "version": 1
    }
    
    response = requests.post(f"{BASE_URL}/update-model", json=update_data)
    print(f"\nUpdate model endpoint:")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200

if __name__ == "__main__":
    print("Testing ML Service API...\n")
    
    try:
        test_health()
        test_predict()
        test_update_model()
        
        # Test predict again to verify model still works after update
        print("\nTesting predict after model update:")
        test_predict()
        
        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to service. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")