from pydantic import BaseModel, Field
from typing import List

class PredictB64Request(BaseModel):
    """Request model for base64 image prediction."""
    image_base64: str = Field(..., description="Base64-encoded image string", min_length=1)

class PredictResponse(BaseModel):
    """Response model for pet breed prediction."""
    label: str = Field(..., description="Predicted pet breed")
    probability: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)
    topk_labels: List[str] = Field(..., description="Top-k predicted breeds")
    topk_probs: List[float] = Field(..., description="Top-k confidence scores")
    
    class Config:
        json_schema_extra = {
            "example": {
                "label": "Golden Retriever",
                "probability": 0.95,
                "topk_labels": ["Golden Retriever", "Labrador Retriever", "German Shepherd"],
                "topk_probs": [0.95, 0.03, 0.02]
            }
        }