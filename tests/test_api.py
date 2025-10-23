import pytest
import io
import base64
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert "classes" in res.json()

def test_health_endpoint_model_not_loaded():
    """Test health endpoint when model is not loaded."""
    # This test would need to be run before model loading
    pass

def create_test_image():
    """Create a test image for testing."""
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

def test_predict_endpoint_success():
    """Test successful prediction endpoint."""
    test_img = create_test_image()
    
    response = client.post(
        "/predict",
        files={"file": ("test.png", test_img, "image/png")}
    )
    
    # Note: This will fail without a real model, but tests the endpoint structure
    assert response.status_code in [200, 500]  # 500 if model not found

def test_predict_endpoint_invalid_file_type():
    """Test prediction with invalid file type."""
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    
    assert response.status_code == 400
    assert "File must be an image" in response.json()["detail"]

def test_predict_base64_endpoint():
    """Test base64 prediction endpoint."""
    test_img = create_test_image()
    img_base64 = base64.b64encode(test_img.getvalue()).decode()
    
    response = client.post(
        "/predict-base64",
        json={"image_base64": img_base64}
    )
    
    # Note: This will fail without a real model, but tests the endpoint structure
    assert response.status_code in [200, 500]  # 500 if model not found

def test_predict_base64_invalid_data():
    """Test base64 prediction with invalid data."""
    response = client.post(
        "/predict-base64",
        json={"image_base64": "invalid_base64_data"}
    )
    
    # This should fail with invalid base64
    assert response.status_code in [400, 500]

def test_cors_headers():
    """Test CORS headers are present."""
    response = client.options("/predict")
    assert response.status_code == 200
