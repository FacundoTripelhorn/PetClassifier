# Pet Breed Classifier

A FastAPI + FastAI application for classifying pet breeds using deep learning models.

## Features

- 🐾 **Pet Breed Classification**: Classify dog and cat breeds from images
- 🚀 **Multiple Inference Types**: 
  - Base inference
  - Test-Time Augmentation (TTA)
  - Mix inference with filtering
  - Ensemble predictions
- 📊 **Model Management**: Discover and manage multiple models
- 🎨 **Streamlit UI**: User-friendly web interface
- 🔌 **REST API**: FastAPI backend with automatic documentation
- 📈 **Model Metadata**: Track model performance and metadata

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PetClassifier
```

2. Create a virtual environment:
```bash
conda create -n petclassifier python=3.11
conda activate petclassifier
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

## Configuration

Create a `.env` file in the root directory (optional):

```env
MODEL_PATH=models/pet_classifier.pkl
MODEL_REPO=https://huggingface.co/Faculo/petclassifier/
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:8501
```

## Usage

### Running the API Server

Start the FastAPI server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Running the Streamlit UI

Start the Streamlit interface:

```bash
streamlit run ui/dashboard.py
```

Or if you have a simple dashboard:

```bash
streamlit run ui/simple_dashboard.py
```

### API Endpoints

- `GET /health` - Health check
- `GET /models` - List available models
- `GET /models/metadata` - Get metadata for all models
- `GET /models/metadata/{model_path}` - Get metadata for a specific model
- `POST /predict` - Predict pet breed from image
- `POST /predict/compare` - Compare predictions from all inference types

### Example API Request

```bash
curl -X POST "http://localhost:8000/predict?inference_type=mix&topk=5" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@pet_image.jpg"
```

## Project Structure

```
PetClassifier/
├── app/                 # FastAPI application
│   ├── api/            # API routes
│   ├── inference/      # Inference strategies
│   ├── models/         # Model metadata management
│   ├── utils/          # Utility functions
│   ├── main.py         # FastAPI app entry point
│   └── settings.py     # Configuration
├── ui/                 # Streamlit UI
│   ├── components/     # UI components
│   ├── api_client.py   # API client
│   ├── config.py       # UI configuration
│   └── ...
├── utils/              # Shared utilities
│   └── hf_sync.py     # HuggingFace model sync
├── models/             # Model files directory
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project configuration
└── README.md           # This file
```

## Model Management

Models are automatically synced from HuggingFace on startup. Place model files in the `models/` directory or configure `MODEL_REPO` to sync from HuggingFace.

Model metadata JSON files can be placed alongside `.pkl` files for additional information about the models.

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
ruff check .
```

## License

MIT

## Author

FacundoTripelhorn

