from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MODEL_PATH: str = "models/pet_classifier.pkl"
    MAX_IMAGE_MB: int = 8
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8501,http://127.0.0.1:3000,http://127.0.0.1:8501"
    LOG_LEVEL: str = "INFO"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
