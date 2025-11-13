"""Utility functions for UI."""

from typing import List, Dict
import streamlit as st


def format_confidence(confidence: float) -> str:
    """Format confidence as percentage string.
    
    Args:
        confidence: Confidence value between 0 and 1
        
    Returns:
        Formatted string (e.g., "95.5%")
    """
    return f"{confidence * 100:.1f}%"


def format_predictions(predictions: List[Dict], topk: int = 5) -> str:
    """Format predictions as a readable string.
    
    Args:
        predictions: List of prediction dictionaries
        topk: Number of top predictions to format
        
    Returns:
        Formatted string
    """
    if not predictions:
        return "No predictions"
    
    formatted = []
    for i, pred in enumerate(predictions[:topk], 1):
        label = pred.get("label", "Unknown")
        confidence = pred.get("confidence", 0.0)
        formatted.append(f"{i}. {label} ({format_confidence(confidence)})")
    
    return "\n".join(formatted)


def get_prediction_color(confidence: float) -> str:
    """Get color for prediction based on confidence.
    
    Args:
        confidence: Confidence value between 0 and 1
        
    Returns:
        Color string (green, yellow, or red)
    """
    if confidence >= 0.7:
        return "green"
    elif confidence >= 0.4:
        return "orange"
    else:
        return "red"


def validate_image_file(uploaded_file) -> bool:
    """Validate uploaded image file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        True if valid, False otherwise
    """
    if uploaded_file is None:
        return False
    
    # Check file size (max 8MB)
    max_size = 8 * 1024 * 1024  # 8MB
    if uploaded_file.size > max_size:
        st.error(f"File too large. Maximum size is 8MB, got {uploaded_file.size / 1024 / 1024:.1f}MB")
        return False
    
    # Check file type
    valid_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if uploaded_file.type not in valid_types:
        st.error(f"Invalid file type: {uploaded_file.type}. Supported: {', '.join(valid_types)}")
        return False
    
    return True

