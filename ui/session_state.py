"""Session state management for Streamlit UI."""

import streamlit as st
from typing import Optional


def get_selected_model() -> Optional[str]:
    """Get the currently selected model path from session state."""
    return st.session_state.get("selected_model", None)


def set_selected_model(model_path: Optional[str]) -> None:
    """Set the selected model path in session state."""
    st.session_state["selected_model"] = model_path


def get_selected_inference_type() -> str:
    """Get the currently selected inference type from session state."""
    return st.session_state.get("selected_inference_type", "mix")


def set_selected_inference_type(inference_type: str) -> None:
    """Set the selected inference type in session state."""
    st.session_state["selected_inference_type"] = inference_type


def get_prediction_history() -> list:
    """Get prediction history from session state."""
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []
    return st.session_state["prediction_history"]


def add_to_prediction_history(prediction: dict) -> None:
    """Add a prediction to the history."""
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []
    st.session_state["prediction_history"].append(prediction)
    # Keep only last 50 predictions
    if len(st.session_state["prediction_history"]) > 50:
        st.session_state["prediction_history"] = st.session_state["prediction_history"][-50:]


def clear_prediction_history() -> None:
    """Clear the prediction history."""
    st.session_state["prediction_history"] = []

