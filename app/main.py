import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.settings import settings
from app.utils import validate_image_size, load_pil_from_bytes, load_pil_from_base64
from app.schemas import PredictB64Request, PredictResponse
from app.inference import PetClassifier

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

app = FastAPI(
    title='Pet Breed Classifier API',
    version='1.0.0',
    description='A FastAPI service for classifying pet breeds using FastAI',
    docs_url='/docs',
    redoc_url='/redoc'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(',')],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier: PetClassifier | None = None

@app.on_event('startup')
def load_model():
    global classifier
    log.info('Loading model...')
    classifier = PetClassifier(settings.MODEL_PATH)
    log.info('Model loaded successfully with %d labels', len(classifier.labels))

@app.get('/health', tags=['meta'], summary='Health Check', description='Check if the API and model are ready')
def health():
    """Health check endpoint to verify API and model status."""
    if classifier is None:
        return JSONResponse(
            status_code=503,
            content={'status': 'error', 'message': 'Model not loaded'}
        )
    
    return {
        'status': 'ok',
        'classes': len(classifier.labels),
        'model_loaded': True,
        'is_mock_model': getattr(classifier, 'is_mock', False),
        'api_version': '1.0.0'
    }

@app.post('/predict', response_model=PredictResponse, tags=['predict'], 
          summary='Predict Pet Breed', 
          description='Upload an image file to predict the pet breed')
async def predict_multipart(file: UploadFile = File(..., description="Image file (JPG, PNG, etc.)")):
    """
    Predict the breed of a pet from an uploaded image.
    
    - **file**: Image file to classify (max 8MB)
    
    Returns the predicted breed with confidence score and top-k predictions.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        b = await file.read()
        validate_image_size(b, settings.MAX_IMAGE_MB)
        img = load_pil_from_bytes(b)
        
        log.info(f"Processing image: {file.filename}, size: {len(b)} bytes")
        label, prob, topk_labels, topk_probs = classifier.predict_pet(img)
        return PredictResponse(label=label, probability=prob, topk_labels=topk_labels, topk_probs=topk_probs)
    except ValueError as ve:
        log.warning(f"Validation error: {ve}")
        raise HTTPException(status_code=413, detail=str(ve))
    except Exception as e:
        log.exception('Prediction error')
        raise HTTPException(status_code=500, detail=f'Prediction failed: {str(e)}')

@app.post('/predict-base64', response_model=PredictResponse, tags=['predict'],
          summary='Predict Pet Breed (Base64)', 
          description='Predict pet breed from a base64-encoded image')
def predict_base64(request: PredictB64Request):
    """
    Predict the breed of a pet from a base64-encoded image.
    
    - **image_base64**: Base64-encoded image string
    
    Returns the predicted breed with confidence score and top-k predictions.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        img = load_pil_from_base64(request.image_base64)
        
        log.info(f"Processing base64 image, size: {len(request.image_base64)} chars")
        label, prob, topk_labels, topk_probs = classifier.predict_pet(img)
        return PredictResponse(label=label, probability=prob, topk_labels=topk_labels, topk_probs=topk_probs)
    except ValueError as ve:
        log.warning(f"Validation error: {ve}")
        raise HTTPException(status_code=413, detail=str(ve))
    except Exception as e:
        log.exception('Prediction error')
        raise HTTPException(status_code=500, detail=f'Prediction failed: {str(e)}')