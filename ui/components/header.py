"""Header component for Streamlit UI."""

import streamlit as st


def render_header() -> None:
    """Render header with app title and description."""
    st.title("🐾 Pet Breed Classifier")
    st.markdown(
        """
        Upload an image of a pet to classify its breed using deep learning models.
        """
    )
    st.markdown("---")

