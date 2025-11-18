"""Inference implementation for multitask FastAI models."""

import logging
import pathlib
from typing import Dict, Optional

from fastapi import UploadFile
from fastai.vision.all import PILImage
from fastai.learner import load_learner

from app.inference.classifier import PetClassifier

log = logging.getLogger(__name__)


class MultitaskInference:
    """Inference helper for models that emit multiple classification heads."""

    def __init__(self, model_path: str, metadata_path: Optional[str] = None) -> None:
        """Initialize the classifier with a model.

        Args:
            model_path: Path to the model file
        """
        self.learn = self._load_learner_safe(model_path)
        self.learn.dls.num_workers = 0
        self.species = list(self.learn.dls.vocab[0])
        self.breeds = list(self.learn.dls.vocab[1])

    def _load_learner_safe(self, model_path):
        """Safely load a FastAI learner.

        Args:
            model_path: Path to the model file

        Returns:
            FastAI learner instance

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model fails to load
        """
        try:
            # Convert string to Path if needed
            model_path = pathlib.Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            return load_learner(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    async def predict_pet(self, file: UploadFile, topk: int = 5) -> Dict:
        """Predict pet breed from image.

        Args:
            file: Uploaded image file
            topk: Number of top predictions to return

        Returns:
            Dictionary with prediction results
        """
        contents = await file.read()
        pil_image = PILImage.create(contents)

        dl = self.learn.dls.test_dl([pil_image])

        preds, _ = self.learn.get_preds(dl=dl)

        species_logits, breed_logits = preds

        species_probs = species_logits[0].softmax(dim=0)
        breed_probs = breed_logits[0].softmax(dim=0)

        species_idx = species_probs.argmax().item()
        breed_idx = breed_probs.argmax().item()

        species = self.species[species_idx]
        breed = self.breeds[breed_idx]

        return {
            "prediction": {"species": species, "breed": breed},
            "top_k": {
                "species": {"label": species, "confidence": species_probs.max().item()},
                "breed": {"label": breed, "confidence": breed_probs.max().item()},
            },
        }
