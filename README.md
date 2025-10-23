# 🐾 Pet Breed Classifier

<div align="center">

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37.1-red.svg)
![FastAI](https://img.shields.io/badge/FastAI-2.7.15-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**An intelligent pet breed classification system powered by FastAI and FastAPI**

[🚀 Quick Start](#-quick-start) • [📚 Documentation](#-documentation) • [🎯 Features](#-features) • [🐳 Docker](#-docker)

</div>


## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (recommended: 3.11 or 3.12)
- **pip** or **poetry** for dependency management
- **Git** for cloning the repository
- **8GB+ RAM** recommended for model inference

### 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/PetClassifier.git
   cd PetClassifier
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate (Windows)
   .venv\Scripts\activate
   
   # Activate (Linux/Mac)
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env with your preferred settings
   # Default values work for most use cases
   ```

5. **Add your model**
   ```bash
   # Place your trained FastAI model file at:
   models/pet_classifier.pkl
   
   # Or download a pre-trained model (see Models section)
   ```

### 🏃‍♂️ Running the Application

#### **Option 1: All-in-One (Recommended for Development)**
```bash
# Run both API and UI together
python run.py

# Or using task runner
task run
```

#### **Option 2: Separate Services (Recommended for Production)**
```bash
# Terminal 1 - Start API Server
task serve
# API will be available at http://localhost:8000

# Terminal 2 - Start Web UI
task ui
# UI will be available at http://localhost:8501
```

### 🌐 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Web UI** | http://localhost:8501 | Main user interface |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **API Server** | http://localhost:8000 | REST API endpoints |
| **Health Check** | http://localhost:8000/health | System status |

### 🎯 First Steps

1. **Open the Web UI** at http://localhost:8501
2. **Upload a pet photo** using the drag-and-drop interface
3. **Click "Identify Breed!"** to get instant results
4. **Explore the detailed analytics** in the tabs below the results
5. **Check out the breed gallery** for inspiration!

## 📚 API Documentation

### 🔍 **Health Check**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Check API and model status |

**Response:**
```json
{
  "status": "ok",
  "classes": 37,
  "model_loaded": true,
  "is_mock_model": false,
  "api_version": "1.0.0"
}
```

### 🎯 **Prediction Endpoints**

#### **File Upload Prediction**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Upload image file for classification |

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body**: Image file (JPG, PNG, etc.)
- **Max Size**: 8MB (configurable)

**Response:**
```json
{
  "label": "Golden Retriever",
  "probability": 0.95,
  "topk_labels": ["Golden Retriever", "Labrador Retriever", "German Shepherd"],
  "topk_probs": [0.95, 0.03, 0.02]
}
```

#### **Base64 Prediction**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict-base64` | Send base64-encoded image |

**Request:**
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### 📖 **Interactive Documentation**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Testing

### **Run Tests**
```bash
# Run all tests
task test

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_api.py

# Run with verbose output
pytest -v tests/
```

### **Test Coverage**
- **API Endpoints**: All endpoints tested
- **Error Handling**: Comprehensive error scenarios
- **Model Integration**: Mock and real model testing
- **Data Validation**: Input validation testing

## 🐳 Docker

### **Build and Run**
```bash
# Build Docker image
task dockerbuild

# Run container
task dockerrun

# Run with custom port
docker run -p 8000:8000 -p 8501:8501 petclassifier
```

### **Docker Compose** (Optional)
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/pet_classifier.pkl
  
  ui:
    build: .
    command: streamlit run ui/dashboard.py
    ports:
      - "8501:8501"
    depends_on:
      - api
```

## 📁 Project Structure

```
PetClassifier/
├── 📁 app/                    # FastAPI Backend
│   ├── 📄 main.py            # Main API application & endpoints
│   ├── 📄 inference.py       # Model inference & prediction logic
│   ├── 📄 schemas.py         # Pydantic data models & validation
│   ├── 📄 settings.py        # Configuration management
│   └── 📄 utils.py           # Utility functions & helpers
├── 📁 ui/                    # Streamlit Frontend
│   ├── 📄 dashboard.py       # Main web interface
│   ├── 📄 styles.css         # Custom CSS styling
│   └── 📄 README.md          # UI documentation
├── 📁 tests/                 # Test Suite
│   ├── 📄 test_api.py        # API endpoint tests
│   └── 📄 README.md          # Testing documentation
├── 📁 models/                # Model Storage
│   ├── 📄 pet_classifier.pkl # Trained FastAI model
│   └── 📄 README.md          # Model documentation
├── 📄 run.py                 # Application launcher
├── 📄 requirements.txt       # Python dependencies
├── 📄 pyproject.toml         # Project configuration
├── 📄 .env.example          # Environment variables template
├── 📄 .dockerfile           # Docker configuration
├── 📄 .gitignore            # Git ignore rules
└── 📄 README.md             # This file
```

## 🔧 Configuration

### **Environment Variables**
Create a `.env` file based on `.env.example`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/pet_classifier.pkl` | Path to the FastAI model file |
| `MAX_IMAGE_MB` | `8` | Maximum image size in MB |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `API_HOST` | `localhost` | API server host |
| `API_PORT` | `8000` | API server port |
| `UI_PORT` | `8501` | Streamlit UI port |

### **Model Configuration**
- **Supported Formats**: `.pkl` (FastAI export format)
- **Model Size**: Typically 50-200MB
- **Memory Requirements**: 2-4GB RAM for inference
- **Supported Classes**: Cats and dogs (37+ breeds)

## 🚀 Deployment

### **Production Deployment**
```bash
# Using Docker (Recommended)
docker build -t petclassifier .
docker run -d -p 8000:8000 -p 8501:8501 petclassifier

# Using systemd (Linux)
sudo systemctl enable petclassifier
sudo systemctl start petclassifier
```

### **Environment-Specific Settings**
- **Development**: Use default settings with mock model
- **Staging**: Enable detailed logging and testing endpoints
- **Production**: Use production model, disable debug features

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### **Development Setup**
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/PetClassifier.git
cd PetClassifier

# Create a feature branch
git checkout -b feature/amazing-feature

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
pytest tests/
```

### **Contribution Guidelines**
1. **Code Style**: Follow PEP 8 and use `ruff` for linting
2. **Testing**: Add tests for new features
3. **Documentation**: Update READMEs and docstrings
4. **Commits**: Use clear, descriptive commit messages
5. **Pull Requests**: Provide detailed descriptions

### **Areas for Contribution**
- 🎨 **UI/UX Improvements**: Better visualizations, mobile responsiveness
- 🤖 **Model Enhancements**: Better accuracy, new breeds
- ⚡ **Performance**: Faster inference, caching
- 🧪 **Testing**: More test coverage, integration tests
- 📚 **Documentation**: Tutorials, examples, guides

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAI** for the amazing deep learning framework
- **FastAPI** for the high-performance web framework
- **Streamlit** for the beautiful web interface
- **Plotly** for interactive data visualizations
- **The pet community** for inspiration and feedback

---

<div align="center">

**Made with ❤️ for pet lovers everywhere**

[⭐ Star this repo](https://github.com/yourusername/PetClassifier) • [🐛 Report a bug](https://github.com/yourusername/PetClassifier/issues) • [💡 Request a feature](https://github.com/yourusername/PetClassifier/issues)

</div>