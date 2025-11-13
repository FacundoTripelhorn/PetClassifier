"""Ensemble inference implementation."""

from fastapi import UploadFile
import pathlib
from app.inference.base_inference import BaseInference
from app.utils.mix_inference import combine_predictions


class EnsembleInference:
    """Ensemble inference class using multiple models."""
    
    def __init__(self):
        """Initialize ensemble inference."""
        self.models_dir = pathlib.Path("models")
        self.classifiers = []
        self._load_models()
    
    def _load_models(self):
        """Load all available models."""
        if not self.models_dir.exists():
            return
        
        # Find all .pkl files
        model_files = list(self.models_dir.rglob("*.pkl"))
        
        for model_file in model_files:
            try:
                classifier = BaseInference(str(model_file))
                self.classifiers.append(classifier)
            except Exception as e:
                # Skip models that fail to load
                continue
    
    async def predict_pet(self, file: UploadFile, topk: int = 5):
        """Predict pet breed from image using ensemble of models.
        
        Args:
            file: Uploaded image file
            topk: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        if not self.classifiers:
            raise RuntimeError("No models available for ensemble inference")
        
        # Get predictions from all models
        prediction_lists = []
        for classifier in self.classifiers:
            try:
                result = await classifier.predict_pet(file, topk=topk)
                topk_list = result.get("top_k", [])
                if topk_list:
                    prediction_lists.append(topk_list)
            except Exception as e:
                # Skip models that fail
                continue
        
        if not prediction_lists:
            raise RuntimeError("All models failed to make predictions")
        
        # Combine predictions
        combined = combine_predictions(prediction_lists, topk=topk)
        
        # Get top prediction
        if combined:
            top_prediction = combined[0]
        else:
            top_prediction = {
                "label": "unknown",
                "confidence": 0.0
            }
        
        return {
            "prediction": {
                "label": top_prediction.get("label", "unknown"),
                "confidence": top_prediction.get("confidence", 0.0)
            },
            "top_k": combined[:topk],
            "num_classes": len(self.classifiers[0].classifier.labels) if self.classifiers else 0,
            "ensemble_size": len(self.classifiers),
            "models_used": len(prediction_lists)
        }

