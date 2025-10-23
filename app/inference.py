from fastai.vision.all import load_learner, PILImage
from typing import Tuple, List
from PIL import Image

class PetClassifier:
    def __init__(self, model_path: str):
        self.learn = load_learner(model_path)
        self.labels = list(self.learn.dls.vocab)

    def predict_pet(self, img: Image.Image, topk: int = 3) -> Tuple[str, float, List[str], List[float]]:
        pred, pred_idx, probs = self.learn.predict(PILImage.create(img))
        topk_probs, topk_idxs = probs.topk(min(topk, len(probs)))
        topk_labels = [self.labels[i] for i in topk_idxs.tolist()]
        return (
            str(pred),
            float(probs[pred_idx]),
            topk_labels,
            [float(p) for p in topk_probs.tolist()]
        )