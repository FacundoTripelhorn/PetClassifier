from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MODEL_PATH: str = "models/pet_classifier.pkl"
    MAX_IMAGE_MB: int = 8
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8501,http://127.0.0.1:3000,http://127.0.0.1:8501"
    LOG_LEVEL: str = "INFO"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    TOPK_MIX: int = 3
    PURITY_THRESHOLD: float = 0.7
    MARGIN_THRESHOLD: float = 0.2
    TTA_N_AUGMENTATIONS: int = 8
    MODEL_REPO: str = "https://huggingface.co/Faculo/petclassifier/"
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()

