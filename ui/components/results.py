"""Results display component for Streamlit UI."""

import streamlit as st
from typing import Dict, List, Optional


def render_results(result: Optional[Dict], show_topk: int = 5) -> None:
    """Render prediction results.
    
    Args:
        result: Prediction result dictionary from API
        show_topk: Number of top predictions to display
    """
    if not result:
        st.warning("No results to display")
        return
    
    prediction = result.get("prediction", {})
    top_predictions = result.get("top_predictions", [])
    
    if prediction:
        # Main prediction
        label = prediction.get("label", "Unknown")
        confidence = prediction.get("confidence", 0.0)
        
        st.success(f"**Predicted:** {label} ({confidence * 100:.1f}%)")
        
        # Top predictions
        if top_predictions:
            st.subheader("Top Predictions")
            for i, pred in enumerate(top_predictions[:show_topk], 1):
                pred_label = pred.get("label", "Unknown")
                pred_conf = pred.get("confidence", 0.0)
                st.write(f"{i}. {pred_label}: {pred_conf * 100:.1f}%")
    
    # Metadata
    if "inference_type" in result:
        st.caption(f"Inference type: {result['inference_type']}")
    if "model_path" in result:
        st.caption(f"Model: {result['model_path']}")

