"""API routes for model metadata."""

from fastapi import APIRouter, HTTPException, Path
from typing import List, Dict
import pathlib
from app.models.metadata import load_metadata, create_default_metadata, ModelMetadata
from app.inference.classifier import PetClassifier

router = APIRouter(prefix="/models", tags=["models"])

MODELS_DIR = pathlib.Path("models")


def discover_models() -> List[Dict[str, str]]:
    """Discover all .pkl model files recursively in the models directory."""
    models = []
    
    if not MODELS_DIR.exists():
        return models
    
    # Find all .pkl files recursively
    for pkl_file in MODELS_DIR.rglob("*.pkl"):
        # Get relative path from models directory
        relative_path = pkl_file.relative_to(MODELS_DIR)
        path_str = str(relative_path).replace("\\", "/")  # Normalize path separators
        
        # Create a display name
        display_name = path_str.replace(".pkl", "").replace("_", " ").title()
        
        models.append({
            "path": path_str,
            "full_path": str(pkl_file),
            "display_name": display_name,
            "name": pkl_file.stem,
            "folder": str(relative_path.parent) if relative_path.parent != pathlib.Path(".") else "root"
        })
    
    # Sort by folder, then by name
    models.sort(key=lambda x: (x["folder"], x["name"]))
    
    return models


@router.get(
    "/metadata",
    summary="Get metadata for all models",
    description="Retrieve metadata for all available model files"
)
def get_all_models_metadata() -> Dict:
    """Get metadata for all models."""
    models = discover_models()
    
    models_with_metadata = []
    for model_info in models:
        model_path = pathlib.Path(model_info["full_path"])
        
        # Try to load metadata
        metadata = load_metadata(model_path)
        
        # If no metadata found, create default
        if metadata is None:
            # Try to get num_classes from the model
            try:
                classifier = PetClassifier(str(model_path))
                num_classes = len(classifier.labels)
            except Exception:
                num_classes = 0
            
            metadata = create_default_metadata(model_path, num_classes)
        
        # Convert to dict and add model info
        metadata_dict = metadata.to_dict()
        metadata_dict.update({
            "path": model_info["path"],
            "display_name": model_info["display_name"],
            "name": model_info["name"],
            "folder": model_info["folder"]
        })
        
        models_with_metadata.append(metadata_dict)
    
    return {
        "models": models_with_metadata,
        "total": len(models_with_metadata)
    }


@router.get(
    "/metadata/{model_path:path}",
    summary="Get metadata for a specific model",
    description="Retrieve metadata for a specific model file by path"
)
def get_model_metadata(model_path: str = Path(..., description="Path to model file (relative to models/)")) -> Dict:
    """Get metadata for a specific model."""
    # Construct full path
    full_path = MODELS_DIR / model_path
    
    # Normalize path separators
    full_path = pathlib.Path(str(full_path).replace("\\", "/"))
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
    
    if not full_path.suffix == ".pkl":
        raise HTTPException(status_code=400, detail=f"File is not a model file (.pkl): {model_path}")
    
    # Try to load metadata
    metadata = load_metadata(full_path)
    
    # If no metadata found, create default
    if metadata is None:
        # Try to get num_classes from the model
        try:
            classifier = PetClassifier(str(full_path))
            num_classes = len(classifier.labels)
        except Exception:
            num_classes = 0
        
        metadata = create_default_metadata(full_path, num_classes)
    
    # Convert to dict
    metadata_dict = metadata.to_dict()
    metadata_dict["path"] = model_path
    
    return metadata_dict

