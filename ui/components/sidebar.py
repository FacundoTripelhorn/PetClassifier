"""Sidebar component for Streamlit UI."""

import streamlit as st
from typing import List, Dict, Optional
from ui.api_client import fetch_models
from ui.session_state import (
    get_selected_model,
    set_selected_model,
    get_selected_inference_type,
    set_selected_inference_type,
)


def render_sidebar() -> None:
    """Render sidebar with model and inference type selection."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        st.subheader("🤖 Model")
        models = fetch_models()
        
        if models:
            model_options = ["Default (auto-select)"]
            model_paths = {"Default (auto-select)": None}
            
            for model in models:
                display_name = model.get("name", model.get("path", "Unknown"))
                if model.get("folder") and model["folder"] != "root":
                    display_name = f"{model['folder']}/{display_name}"
                model_options.append(display_name)
                model_paths[display_name] = model.get("path")
            
            selected_display = st.selectbox(
                "Select Model",
                options=model_options,
                index=0,
                help="Choose which model to use"
            )
            
            selected_path = model_paths.get(selected_display)
            set_selected_model(selected_path)
        else:
            st.warning("No models available")
            set_selected_model(None)
        
        # Inference type selection
        st.subheader("🔬 Inference Type")
        inference_types = {
            "base": "Base",
            "tta": "TTA (Test-Time Augmentation)",
            "mix": "Mix",
            "ensemble": "Ensemble"
        }
        
        current_type = get_selected_inference_type()
        selected_type = st.selectbox(
            "Select Inference Type",
            options=list(inference_types.keys()),
            format_func=lambda x: inference_types[x],
            index=list(inference_types.keys()).index(current_type) if current_type in inference_types else 0,
            help="Choose inference method"
        )
        set_selected_inference_type(selected_type)
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.info(
            "This app uses deep learning models to classify pet breeds. "
            "Select different models and inference types to compare results."
        )

