"""Mix inference utilities for combining predictions."""

from typing import List, Dict
from app.settings import settings


def filter_predictions_by_purity(
    predictions: List[Dict],
    purity_threshold: float = None
) -> List[Dict]:
    """Filter predictions based on purity threshold.
    
    Purity is the confidence difference between top prediction and second prediction.
    Higher purity means more confident predictions.
    
    Args:
        predictions: List of prediction dictionaries with 'confidence' key
        purity_threshold: Minimum purity required (uses settings.PURITY_THRESHOLD if None)
        
    Returns:
        Filtered list of predictions
    """
    if purity_threshold is None:
        purity_threshold = settings.PURITY_THRESHOLD
    
    if len(predictions) < 2:
        return predictions
    
    # Calculate purity (difference between top and second)
    top_confidence = predictions[0].get("confidence", 0.0)
    second_confidence = predictions[1].get("confidence", 0.0) if len(predictions) > 1 else 0.0
    purity = top_confidence - second_confidence
    
    if purity >= purity_threshold:
        return predictions
    else:
        return []


def filter_predictions_by_margin(
    predictions: List[Dict],
    margin_threshold: float = None
) -> List[Dict]:
    """Filter predictions based on margin threshold.
    
    Margin is the confidence of the top prediction.
    Higher margin means more confident top prediction.
    
    Args:
        predictions: List of prediction dictionaries with 'confidence' key
        margin_threshold: Minimum margin required (uses settings.MARGIN_THRESHOLD if None)
        
    Returns:
        Filtered list of predictions
    """
    if margin_threshold is None:
        margin_threshold = settings.MARGIN_THRESHOLD
    
    if not predictions:
        return []
    
    top_confidence = predictions[0].get("confidence", 0.0)
    
    if top_confidence >= margin_threshold:
        return predictions
    else:
        return []


def combine_predictions(
    prediction_lists: List[List[Dict]],
    topk: int = None
) -> List[Dict]:
    """Combine multiple prediction lists into a single ranked list.
    
    Args:
        prediction_lists: List of prediction lists to combine
        topk: Number of top predictions to return (uses settings.TOPK_MIX if None)
        
    Returns:
        Combined and ranked list of predictions
    """
    if topk is None:
        topk = settings.TOPK_MIX
    
    # Aggregate predictions by label
    label_scores = {}
    label_counts = {}
    
    for pred_list in prediction_lists:
        for pred in pred_list:
            label = pred.get("label", "")
            confidence = pred.get("confidence", 0.0)
            
            if label not in label_scores:
                label_scores[label] = 0.0
                label_counts[label] = 0
            
            label_scores[label] += confidence
            label_counts[label] += 1
    
    # Calculate average confidence for each label
    combined = []
    for label, total_score in label_scores.items():
        count = label_counts[label]
        avg_confidence = total_score / count if count > 0 else 0.0
        combined.append({
            "label": label,
            "confidence": avg_confidence,
            "count": count
        })
    
    # Sort by confidence (descending)
    combined.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Return topk
    return combined[:topk]

