"""Model analytics component."""

import streamlit as st
from typing import List, Dict


def render_model_analytics(prediction_history: List[Dict]) -> None:
    """Render analytics from prediction history.
    
    Args:
        prediction_history: List of past predictions
    """
    if not prediction_history:
        st.info("No prediction history available")
        return
    
    st.subheader("📈 Analytics")
    
    # Count predictions by inference type
    inference_counts = {}
    for pred in prediction_history:
        inf_type = pred.get("inference_type", "unknown")
        inference_counts[inf_type] = inference_counts.get(inf_type, 0) + 1
    
    if inference_counts:
        st.bar_chart(inference_counts)

