from fastai.vision.all import load_learner, PILImage
from typing import Tuple, List, Union
from PIL import Image
from io import BytesIO
import numpy as np
from pathlib import Path

class PetClassifier:
    def __init__(self, model_path_str):
        self.learn = self._load_learner_safe(model_path_str)
        self.labels = list(self.learn.dls.vocab)
    
    def _load_learner_safe(self, model_path):
        try:
            # Convert string to Path if needed
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            return load_learner(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def predict_pet(self, img: Union[Image.Image, bytes], topk: int = 3) -> Tuple[str, float, List[str], List[float]]:
        """
        Predict pet breed using FastAI's standard predict method.
        
        Args:
            img: PIL Image or bytes to classify
            topk: Number of top predictions to return
            
        Returns:
            Tuple of (predicted_label, confidence, topk_labels, topk_probabilities)
        """
        # Handle bytes input
        if isinstance(img, bytes):
            img = Image.open(BytesIO(img)).convert('RGB')
        
        # Ensure PIL Image is RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert PIL Image to numpy array, then to FastAI PILImage
        # This is the proper way per FastAI docs
        img_array = np.array(img)
        pil_img = PILImage.create(img_array)
        
        # Use FastAI's standard predict method
        pred, pred_idx, probs = self.learn.predict(pil_img)
        
        # Get top-k predictions
        topk_probs, topk_idxs = probs.topk(min(topk, len(probs)))
        topk_labels = [self.labels[i.item()] for i in topk_idxs]
        
        return (
            str(pred),
            float(probs[pred_idx]),
            topk_labels,
            [float(p) for p in topk_probs.tolist()]
        )