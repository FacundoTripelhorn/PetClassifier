"""API client for communicating with the FastAPI backend."""

import requests
from typing import List, Dict, Optional
from PIL import Image
import io
from ui.config import (
    API_BASE_URL,
    MODELS_URL,
    MODELS_METADATA_URL,
    PREDICT_URL,
    COMPARE_URL,
    MODELS_TIMEOUT,
    PREDICT_TIMEOUT,
)


def fetch_models() -> List[Dict]:
    """Fetch available models from the API.
    
    Returns:
        List of model dictionaries, or empty list if request fails
    """
    try:
        response = requests.get(MODELS_URL, timeout=MODELS_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            return data.get("models", [])
    except Exception as e:
        print(f"Error fetching models: {e}")
    return []


def fetch_models_metadata() -> List[Dict]:
    """Fetch models with metadata from the API.
    
    Returns:
        List of model dictionaries with metadata, or empty list if request fails
    """
    try:
        response = requests.get(f"{API_BASE_URL}/models/metadata", timeout=MODELS_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            return data.get("models", [])
    except Exception as e:
        print(f"Error fetching models metadata: {e}")
    return []


def predict_image(
    image: Image.Image,
    model_path: Optional[str],
    inference_type: str,
    topk: int = 5
) -> Optional[Dict]:
    """Send image to API for prediction.
    
    Args:
        image: PIL Image object
        model_path: Path to model file (optional)
        inference_type: Type of inference (base, tta, mix, ensemble)
        topk: Number of top predictions to return
        
    Returns:
        Prediction result dictionary, or None if request fails
    """
    try:
        # Convert image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        
        # Prepare request
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        params = {
            "inference_type": inference_type,
            "topk": topk
        }
        if model_path:
            params["model_path"] = model_path
        
        response = requests.post(
            PREDICT_URL,
            files=files,
            params=params,
            timeout=PREDICT_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Prediction failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error predicting image: {e}")
    return None


def compare_all_types(
    image: Image.Image,
    model_path: Optional[str],
    topk: int
) -> Optional[Dict]:
    """Compare predictions from all inference types.
    
    Args:
        image: PIL Image object
        model_path: Path to model file (optional)
        topk: Number of top predictions to return
        
    Returns:
        Comparison result dictionary, or None if request fails
    """
    try:
        # Convert image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        
        # Prepare request
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        params = {"topk": topk}
        if model_path:
            params["model_path"] = model_path
        
        response = requests.post(
            COMPARE_URL,
            files=files,
            params=params,
            timeout=PREDICT_TIMEOUT * 4  # Longer timeout for comparison
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Comparison failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error comparing inference types: {e}")
    return None


def check_api_health() -> bool:
    """Check if the API is healthy.
    
    Returns:
        True if API is healthy, False otherwise
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=MODELS_TIMEOUT)
        return response.status_code == 200
    except Exception:
        return False

