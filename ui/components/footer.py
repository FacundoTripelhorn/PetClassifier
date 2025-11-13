"""Footer component for Streamlit UI."""

import streamlit as st


def render_footer() -> None:
    """Render footer with app information."""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Pet Breed Classifier - Powered by FastAI & FastAPI</p>
        </div>
        """,
        unsafe_allow_html=True
    )

