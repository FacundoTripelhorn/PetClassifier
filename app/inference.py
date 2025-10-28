from fastai.vision.all import load_learner
import pathlib

pathlib.PosixPath = pathlib.WindowsPath

class PetClassifier:
    def __init__(self, model_path_str):
        self.learn = self._load_learner_safe(model_path_str)
        self.labels = list(self.learn.dls.vocab)
    
    def _load_learner_safe(self, model_path):
        try:
            # Convert string to Path if needed
            model_path = pathlib.Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            return load_learner(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")