"""Instructions component for Streamlit UI."""

import streamlit as st


def render_instructions() -> None:
    """Render instructions for using the app."""
    with st.expander("📖 How to use", expanded=False):
        st.markdown("""
        1. **Upload an image** of a pet (dog or cat)
        2. **Select a model** (optional - uses default if not specified)
        3. **Choose inference type**:
           - **Base**: Standard prediction
           - **TTA**: Test-Time Augmentation (more accurate, slower)
           - **Mix**: Combines multiple predictions with filtering
           - **Ensemble**: Uses multiple models
        4. **Click Predict** to get breed classification
        5. View top predictions with confidence scores
        """)

