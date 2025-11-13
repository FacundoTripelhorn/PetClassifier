"""Test-Time Augmentation (TTA) inference implementation."""

from fastapi import UploadFile
from fastai.vision.all import PILImage
import torch
from app.inference.classifier import PetClassifier
from app.settings import settings


class TTAInference:
    """Test-Time Augmentation inference class."""
    
    def __init__(self, model_path: str):
        """Initialize TTA inference with a model.
        
        Args:
            model_path: Path to the model file
        """
        self.classifier = PetClassifier(model_path)
        self.n_augmentations = settings.TTA_N_AUGMENTATIONS
    
    async def predict_pet(self, file: UploadFile, topk: int = 5):
        """Predict pet breed from image using Test-Time Augmentation.
        
        Args:
            file: Uploaded image file
            topk: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        contents = await file.read()
        pil_image = PILImage.create(contents)
        
        # Get base prediction
        predictions = []
        
        # Original image
        pred_label, _, probs = self.classifier.learn.predict(pil_image)
        predictions.append(probs)
        
        # Augmented predictions
        for _ in range(self.n_augmentations - 1):
            # Apply random augmentation
            aug_image = self._augment_image(pil_image)
            _, _, probs = self.classifier.learn.predict(aug_image)
            predictions.append(probs)
        
        # Average probabilities
        avg_probs = torch.stack(predictions).mean(dim=0)
        
        # Get top predictions
        k = max(1, min(int(topk or 5), len(self.classifier.labels)))
        top_probs, top_indices = avg_probs.topk(k)
        
        # Get predicted label (highest probability)
        predicted_idx = avg_probs.argmax().item()
        predicted_label = self.classifier.labels[predicted_idx]
        
        top_k = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            label = self.classifier.labels[int(idx)]
            top_k.append({
                "label": str(label),
                "confidence": float(prob)
            })
        
        return {
            "prediction": {
                "label": str(predicted_label),
                "confidence": float(avg_probs.max().item())
            },
            "top_k": top_k,
            "num_classes": len(self.classifier.labels),
            "tta_augmentations": self.n_augmentations
        }
    
    def _augment_image(self, image):
        """Apply random augmentation to image."""
        # FastAI's PILImage can be augmented using torchvision transforms
        # For now, return the original image as TTA augmentation
        # In a production system, you'd apply random flips, rotations, etc.
        import random
        from PIL import Image
        
        # Simple augmentation: random horizontal flip
        if random.random() > 0.5:
            return PILImage.create(image.transpose(Image.FLIP_LEFT_RIGHT))
        return image

