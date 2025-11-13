"""Performance metrics component."""

import streamlit as st
from typing import List, Dict


def render_performance_metrics(prediction_history: List[Dict]) -> None:
    """Render performance metrics from prediction history.
    
    Args:
        prediction_history: List of past predictions
    """
    if not prediction_history:
        st.info("No performance data available")
        return
    
    st.subheader("⚡ Performance Metrics")
    
    # Calculate average confidence
    confidences = []
    for pred in prediction_history:
        pred_data = pred.get("prediction", {})
        conf = pred_data.get("confidence", 0.0)
        if conf > 0:
            confidences.append(conf)
    
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        st.metric("Average Confidence", f"{avg_confidence * 100:.1f}%")
        st.metric("Total Predictions", len(prediction_history))

