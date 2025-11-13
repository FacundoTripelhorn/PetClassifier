"""Mix inference implementation with filtering."""

from fastapi import UploadFile
from app.inference.base_inference import BaseInference
from app.inference.tta_inference import TTAInference
from app.utils.mix_inference import (
    filter_predictions_by_purity,
    filter_predictions_by_margin,
    combine_predictions
)
from app.settings import settings


class MixInference:
    """Mix inference class combining base and TTA with filtering."""
    
    def __init__(self, model_path: str):
        """Initialize mix inference with a model.
        
        Args:
            model_path: Path to the model file
        """
        self.base_inference = BaseInference(model_path)
        self.tta_inference = TTAInference(model_path)
    
    async def predict_pet(self, file: UploadFile, topk: int = 5):
        """Predict pet breed from image using mix inference.
        
        Combines base and TTA predictions with filtering.
        
        Args:
            file: Uploaded image file
            topk: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        # Get predictions from both methods
        base_result = await self.base_inference.predict_pet(file, topk=topk)
        tta_result = await self.tta_inference.predict_pet(file, topk=topk)
        
        # Extract top_k lists
        base_topk = base_result.get("top_k", [])
        tta_topk = tta_result.get("top_k", [])
        
        # Filter base predictions
        base_filtered = filter_predictions_by_purity(base_topk)
        base_filtered = filter_predictions_by_margin(base_filtered)
        
        # Filter TTA predictions
        tta_filtered = filter_predictions_by_purity(tta_topk)
        tta_filtered = filter_predictions_by_margin(tta_filtered)
        
        # Combine filtered predictions
        prediction_lists = []
        if base_filtered:
            prediction_lists.append(base_filtered)
        if tta_filtered:
            prediction_lists.append(tta_filtered)
        
        if not prediction_lists:
            # If both filtered out, use base prediction
            combined = base_topk[:topk] if base_topk else tta_topk[:topk]
        else:
            combined = combine_predictions(prediction_lists, topk=topk)
        
        # Get top prediction
        if combined:
            top_prediction = combined[0]
        else:
            # Fallback to base prediction
            top_prediction = base_result.get("prediction", {
                "label": "unknown",
                "confidence": 0.0
            })
        
        return {
            "prediction": {
                "label": top_prediction.get("label", "unknown"),
                "confidence": top_prediction.get("confidence", 0.0)
            },
            "top_k": combined[:topk],
            "num_classes": base_result.get("num_classes", 0),
            "base_filtered": len(base_filtered) if base_filtered else 0,
            "tta_filtered": len(tta_filtered) if tta_filtered else 0
        }

