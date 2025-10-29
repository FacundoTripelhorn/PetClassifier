from fastai.vision.all import load_learner, PILImage
from fastapi import UploadFile, File
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
            
    async def predict_pet(self, file: UploadFile = File(...), topk: int = 5):
        contents = await file.read()
        pil_image = PILImage.create(contents)

        # FastAI returns: (predicted_label, predicted_index_tensor, probabilities_tensor)
        predicted_label, _, probabilities = self.learn.predict(pil_image)

        # Clamp topk to available labels
        k = max(1, min(int(topk or 5), len(self.labels)))
        top_probs, top_indices = probabilities.topk(k)

        top_k = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            label = self.labels[int(idx)]
            top_k.append({
                "label": str(label),
                "confidence": float(prob)
            })

        return {
            "prediction": {
                "label": str(predicted_label),
                "confidence": float(probabilities.max().item())
            },
            "top_k": top_k,
            "num_classes": len(self.labels)
        }
        