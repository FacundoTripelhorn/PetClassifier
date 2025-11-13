"""UI configuration constants."""

API_BASE_URL = "http://localhost:8000"
MODELS_URL = f"{API_BASE_URL}/models"
MODELS_METADATA_URL = f"{API_BASE_URL}/models/metadata"
PREDICT_URL = f"{API_BASE_URL}/predict"
COMPARE_URL = f"{API_BASE_URL}/predict/compare"
HEALTH_URL = f"{API_BASE_URL}/health"

MODELS_TIMEOUT = 5
PREDICT_TIMEOUT = 30

