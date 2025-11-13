"""Base inference implementation."""

from fastapi import UploadFile
from app.inference.classifier import PetClassifier


class BaseInference:
    """Base inference class using standard prediction."""
    
    def __init__(self, model_path: str):
        """Initialize base inference with a model.
        
        Args:
            model_path: Path to the model file
        """
        self.classifier = PetClassifier(model_path)
    
    async def predict_pet(self, file: UploadFile, topk: int = 5):
        """Predict pet breed from image.
        
        Args:
            file: Uploaded image file
            topk: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        return await self.classifier.predict_pet(file, topk=topk)

