import logging
import pathlib
from contextlib import asynccontextmanager
from typing import Dict, Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.inference.factory import InferenceFactory
from app.settings import settings
from app.api.models import router as models_router
from utils.hf_sync import sync_hf_models

pathlib.PosixPath = pathlib.WindowsPath

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    sync_hf_models(settings.MODEL_REPO, patterns=("*.pkl","*.json"), prune=True)
    yield


app = FastAPI(
    title="Pet Breed Classifier API",
    version="1.0.0",
    description="A FastAPI service for classifying pet breeds using FastAI",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include model metadata router
app.include_router(models_router)

# Store all classifiers (cached by model_path + inference_type)
classifiers_cache: Dict[str, object] = {}

# Models directory
MODELS_DIR = pathlib.Path("models")


def discover_models() -> List[Dict[str, str]]:
    """Discover all .pkl model files recursively in the models directory."""
    models = []
    
    if not MODELS_DIR.exists():
        log.warning(f"Models directory not found: {MODELS_DIR}")
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


def get_classifier(model_path: str, inference_type: str):
    """Get or create a classifier for the given model path and inference type.
    
    Uses the InferenceFactory pattern for cleaner code.
    """
    cache_key = f"{model_path}:{inference_type}"
    
    # Check cache first
    if cache_key in classifiers_cache:
        return classifiers_cache[cache_key]
    
    # Create full path for non-ensemble types
    if inference_type != "ensemble":
        # Validate model_path is not empty or just a directory name
        if not model_path or model_path.strip() == "":
            raise ValueError("model_path cannot be empty")
        
        # Remove leading slash if present
        model_path = model_path.lstrip("/")
        
        full_path = MODELS_DIR / model_path
        
        # Validate it's a file, not a directory
        if not full_path.exists():
            raise FileNotFoundError(f"Model file not found: {full_path}")
        if full_path.is_dir():
            raise ValueError(f"Path is a directory, not a file: {full_path}")
        if not full_path.is_file():
            raise ValueError(f"Path is not a valid file: {full_path}")
        
        model_path_str = str(full_path)
        log.info(f"Loading model from: {model_path_str}")
    else:
        model_path_str = None
    
    # Use factory to create classifier
    try:
        classifier = InferenceFactory.create(
            name=inference_type,
            model_path=model_path_str
        )
    except ValueError as e:
        log.error(f"Failed to create classifier: {e}")
        raise
    
    # Cache it
    classifiers_cache[cache_key] = classifier
    log.info(f"Cached classifier: {cache_key}")
    
    return classifier


@app.on_event("startup")
def startup_event():
    """Startup event - discover available models."""
    log.info("Starting up...")
    models = discover_models()
    log.info(f"Discovered {len(models)} model(s) in {MODELS_DIR}")
    for model in models:
        log.info(f"  - {model['path']} ({model['folder']})")


@app.get(
    "/models",
    tags=["meta"],
    summary="List Available Models",
    description="Get a list of all available model files in the models directory",
)
def list_models():
    """List all available model files."""
    models = discover_models()
    return {
        "models": models,
        "total": len(models),
        "models_dir": str(MODELS_DIR)
    }


@app.get(
    "/health",
    tags=["meta"],
    summary="Health Check",
    description="Check if the API and models are ready",
)
def health():
    """Health check endpoint to verify API and model status."""
    models = discover_models()
    return {
        "status": "ok",
        "models_available": len(models),
        "cached_classifiers": len(classifiers_cache),
        "api_version": "1.0.0",
    }


@app.post(
    "/predict",
    tags=["predict"],
    summary="Predict pet breed from image",
    description="Predict pet breed from image using specified inference type and model",
)
async def predict(
    file: UploadFile = File(...), 
    topk: int = 5,
    inference_type: str = Query("mix", description="Inference type: base, tta, mix, ensemble, or multitask"),
    model_path: Optional[str] = Query(None, description="Path to model file (relative to models/). If not provided, uses default.")
):
    """Predict pet breed using the specified inference type and model."""
    # Validate inference type
    valid_types = ["base", "tta", "mix", "ensemble", "multitask"]
    if inference_type not in valid_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid inference type '{inference_type}'. Available: {', '.join(valid_types)}"
        )
    
    # For ensemble, model_path is not used
    if inference_type == "ensemble":
        model_path = "ensemble"  # Placeholder for cache key
    
    # Use default model if not specified
    if model_path is None:
        # Try to use the default model from settings, or first available model
        default_path = settings.MODEL_PATH
        default_path_obj = pathlib.Path(default_path)
        if default_path_obj.exists() and default_path_obj.is_file():
            try:
                model_path = str(default_path_obj.relative_to(MODELS_DIR))
            except ValueError:
                # If default_path is not relative to MODELS_DIR, use it as-is
                model_path = default_path
        else:
            models = discover_models()
            if not models:
                raise HTTPException(status_code=503, detail="No models available")
            model_path = models[0]["path"]
            log.info(f"Using first available model: {model_path}")
    
    # Validate model_path is set
    if not model_path or model_path.strip() == "":
        raise HTTPException(status_code=400, detail="model_path is required or no models available")
    
    log.info(f"Request: inference_type={inference_type}, model_path={model_path}")
    
    try:
        classifier = get_classifier(model_path, inference_type)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error(f"Error loading classifier: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    result = await classifier.predict_pet(file, topk=topk)

    log.info(f"Prediction result: {result}")
    
    # Add metadata to result
    result["inference_type"] = inference_type
    result["model_path"] = model_path
    
    pred = result.get("prediction", {})
    log.info(
        "[%s:%s] Predicted %s (%.2f%%)",
        inference_type.upper(),
        model_path,
        pred.get("label", "unknown"),
        pred.get("confidence", 0.0) * 100,
    )
    return JSONResponse(content=result)


@app.post(
    "/predict/compare",
    tags=["predict"],
    summary="Compare all inference types",
    description="Run prediction with all available inference types and compare results",
)
async def compare_all(
    file: UploadFile = File(...),
    topk: int = 5,
    inference_types: Optional[str] = Query(None, description="Comma-separated list of inference types to compare. If not provided, compares all available types."),
    model_path: Optional[str] = Query(None, description="Path to model file (relative to models/). If not provided, uses default.")
):
    """Compare predictions from all inference types."""
    # Read file content once
    file_contents = await file.read()
    file_filename = file.filename or "image.jpg"
    file_content_type = file.content_type or "image/jpeg"
    
    # Determine which inference types to use
    valid_types = ["base", "tta", "mix", "ensemble", "multitask"]
    if inference_types:
        types_to_compare = [t.strip() for t in inference_types.split(",")]
        types_to_compare = [t for t in types_to_compare if t in valid_types]
    else:
        types_to_compare = valid_types
    
    if not types_to_compare:
        raise HTTPException(status_code=400, detail="No valid inference types to compare")
    
    # Use default model if not specified
    if model_path is None:
        default_path = settings.MODEL_PATH
        if pathlib.Path(default_path).exists():
            model_path = str(pathlib.Path(default_path).relative_to(MODELS_DIR))
        else:
            models = discover_models()
            if not models:
                raise HTTPException(status_code=503, detail="No models available")
            model_path = models[0]["path"]
    
    results = {}
    errors = {}
    
    # Create a mock UploadFile class
    class MockUploadFile:
        def __init__(self, contents, filename, content_type):
            self._contents = contents
            self.filename = filename
            self.content_type = content_type
        
        async def read(self):
            return self._contents
    
    # Run predictions for each inference type
    for inf_type in types_to_compare:
        try:
            # For ensemble, model_path is not used
            actual_model_path = "ensemble" if inf_type == "ensemble" else model_path
            
            classifier = get_classifier(actual_model_path, inf_type)
            
            mock_file = MockUploadFile(file_contents, file_filename, file_content_type)
            
            result = await classifier.predict_pet(mock_file, topk=topk)
            
            result["inference_type"] = inf_type
            result["model_path"] = actual_model_path if inf_type != "ensemble" else "ensemble/*"
            results[inf_type] = result
            
        except Exception as e:
            log.error(f"Error with {inf_type} inference: {e}")
            errors[inf_type] = str(e)
    
    # Create comparison summary
    comparison = {
        "results": results,
        "errors": errors,
        "summary": {
            "total_types": len(types_to_compare),
            "successful": len(results),
            "failed": len(errors),
            "inference_types_compared": types_to_compare,
            "model_path": model_path
        }
    }
    
    # Add consensus analysis if multiple successful predictions
    if len(results) > 1:
        predictions = [r["prediction"]["label"] for r in results.values()]
        consensus = max(set(predictions), key=predictions.count)
        comparison["consensus"] = {
            "most_common_prediction": consensus,
            "agreement_count": predictions.count(consensus),
            "total_predictions": len(predictions)
        }
    
    return JSONResponse(content=comparison)

