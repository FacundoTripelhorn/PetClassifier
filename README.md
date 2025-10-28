# 🐾 Pet Breed Classifier

<div align="center">

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37.1-red.svg)
![FastAI](https://img.shields.io/badge/FastAI-2.7.15-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**An intelligent pet breed classification system powered by FastAI, FastAPI and Streamlit**

</div>


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