"""Factory for creating inference instances."""

from typing import Optional
from app.inference.base_inference import BaseInference
from app.inference.tta_inference import TTAInference
from app.inference.mix_inference import MixInference
from app.inference.ensemble_inference import EnsembleInference


class InferenceFactory:
    """Factory class for creating inference instances."""
    
    _inference_types = {
        "base": BaseInference,
        "tta": TTAInference,
        "mix": MixInference,
        "ensemble": EnsembleInference,
    }
    
    @classmethod
    def create(cls, name: str, model_path: Optional[str] = None):
        """Create an inference instance by name.
        
        Args:
            name: Name of the inference type (base, tta, mix, ensemble)
            model_path: Path to the model file (required for base, tta, mix)
            
        Returns:
            An inference instance
            
        Raises:
            ValueError: If the inference type is not supported
        """
        if name not in cls._inference_types:
            available = ", ".join(cls._inference_types.keys())
            raise ValueError(f"Unknown inference type '{name}'. Available: {available}")
        
        inference_class = cls._inference_types[name]
        
        # Ensemble doesn't need model_path
        if name == "ensemble":
            return inference_class()
        else:
            if model_path is None:
                raise ValueError(f"model_path is required for inference type '{name}'")
            return inference_class(model_path)

