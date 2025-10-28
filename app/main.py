import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.settings import settings
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

@app.post('/predict', tags=['predict'], summary='Predict pet breed from image', description='Predict pet breed from image')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    pred = classifier.learn.predict(contents)
    log.info(pred)
    return {"prediction": pred}
