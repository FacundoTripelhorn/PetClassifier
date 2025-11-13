"""Model information display component."""

import streamlit as st
from typing import Dict, Optional


def render_model_info(model_metadata: Optional[Dict]) -> None:
    """Display detailed model information.
    
    Args:
        model_metadata: Model metadata dictionary
    """
    if not model_metadata:
        st.info("No model metadata available")
        return
    
    st.markdown("### 📊 Model Information")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Architecture", model_metadata.get("architecture", "Unknown"))
    with col2:
        accuracy = model_metadata.get("accuracy", 0.0)
        st.metric("Accuracy", f"{accuracy:.2%}" if accuracy > 0 else "N/A")
    with col3:
        st.metric("Classes", model_metadata.get("num_classes", 0))
    
    # Additional info
    col4, col5, col6 = st.columns(3)
    with col4:
        model_size = model_metadata.get("model_size_mb", 0.0)
        st.metric("Model Size", f"{model_size:.2f} MB" if model_size > 0 else "N/A")
    with col5:
        epochs = model_metadata.get("epochs", 0)
        st.metric("Epochs", epochs if epochs > 0 else "N/A")
    with col6:
        lr = model_metadata.get("learning_rate", 0.0)
        st.metric("Learning Rate", f"{lr:.6f}" if lr > 0 else "N/A")
    
    # Training details expander
    with st.expander("📝 Training Details", expanded=False):
        st.json(model_metadata)

