"""Model metadata management utilities."""

import json
import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class ModelMetadata(BaseModel):
    """Model metadata schema."""
    
    architecture: str = Field(default="Unknown", description="Model architecture")
    accuracy: float = Field(default=0.0, description="Model accuracy (0.0-1.0)")
    num_classes: int = Field(default=0, description="Number of classes")
    model_size_mb: float = Field(default=0.0, description="Model size in MB")
    epochs: int = Field(default=0, description="Number of training epochs")
    learning_rate: float = Field(default=0.0, description="Learning rate used")
    model_path: Optional[str] = Field(default=None, description="Path to model file")
    description: Optional[str] = Field(default=None, description="Model description")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    
    def to_dict(self) -> dict:
        """Convert metadata to dictionary."""
        return self.model_dump(exclude_none=True)


def get_metadata_path(model_path: Path) -> Path:
    """Get the path to the metadata JSON file for a model.
    
    Args:
        model_path: Path to the .pkl model file
        
    Returns:
        Path to the metadata JSON file
    """
    # Try {model_path}.json first, then {model_path}_metadata.json
    metadata_path = model_path.with_suffix('.json')
    if not metadata_path.exists():
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
    return metadata_path


def load_metadata(model_path: Path) -> Optional[ModelMetadata]:
    """Load metadata from a JSON file associated with a model.
    
    Args:
        model_path: Path to the .pkl model file
        
    Returns:
        ModelMetadata object if found, None otherwise
    """
    metadata_path = get_metadata_path(model_path)
    
    if not metadata_path.exists():
        log.debug(f"Metadata file not found: {metadata_path}")
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure model_path is set
        if 'model_path' not in data:
            data['model_path'] = str(model_path)
        
        return ModelMetadata(**data)
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON in metadata file {metadata_path}: {e}")
        return None
    except Exception as e:
        log.error(f"Error loading metadata from {metadata_path}: {e}")
        return None


def create_default_metadata(model_path: Path, num_classes: int) -> ModelMetadata:
    """Create default metadata for a model.
    
    Args:
        model_path: Path to the .pkl model file
        num_classes: Number of classes in the model
        
    Returns:
        ModelMetadata object with default values
    """
    # Calculate model size if file exists
    model_size_mb = 0.0
    if model_path.exists():
        try:
            model_size_mb = model_path.stat().st_size / (1024 * 1024)  # Convert to MB
        except Exception as e:
            log.warning(f"Could not get model size for {model_path}: {e}")
    
    return ModelMetadata(
        architecture="Unknown",
        accuracy=0.0,
        num_classes=num_classes,
        model_size_mb=model_size_mb,
        epochs=0,
        learning_rate=0.0,
        model_path=str(model_path),
        description=f"Model file: {model_path.name}"
    )

