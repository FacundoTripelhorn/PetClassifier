"""Inference package for pet breed classification."""

from app.inference.classifier import PetClassifier
from app.inference.base_inference import BaseInference
from app.inference.tta_inference import TTAInference
from app.inference.mix_inference import MixInference
from app.inference.ensemble_inference import EnsembleInference
from app.inference.multitask_inference import MultitaskInference
from app.inference.factory import InferenceFactory

__all__ = [
    "PetClassifier",
    "BaseInference",
    "TTAInference",
    "MixInference",
    "EnsembleInference",
    "MultitaskInference",
    "InferenceFactory",
]

