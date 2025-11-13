"""Model comparison component."""

import streamlit as st
import pandas as pd
from typing import List, Dict


def render_model_comparison(models_metadata: List[Dict]) -> None:
    """Render comparison table for multiple models.
    
    Args:
        models_metadata: List of model metadata dictionaries
    """
    if not models_metadata:
        st.info("No model metadata available for comparison")
        return
    
    df = pd.DataFrame(models_metadata)
    
    # Select relevant columns
    display_cols = ["name", "architecture", "accuracy", "num_classes", "model_size_mb"]
    available_cols = [col for col in display_cols if col in df.columns]
    
    if available_cols:
        st.dataframe(df[available_cols], use_container_width=True)
    else:
        st.json(models_metadata)

