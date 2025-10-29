import streamlit as st
import requests
from io import BytesIO
from PIL import Image
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="🐾 Pet Breed Classifier", 
    page_icon="🐕", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load shared CSS from file
css_path = Path(__file__).parent / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# API configuration
API_URL = "http://localhost:8000/predict"

# Initialize session state
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# Header
st.markdown("""
<div class="main-header">
    <h1>🐾 Pet Breed Classifier</h1>
    <p>Upload a photo of your pet to identify its breed with AI!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for settings (optional)
with st.sidebar:
    st.header("⚙️ Settings")
    topk = st.slider("Number of top predictions", min_value=3, max_value=10, value=5)
    st.markdown("---")
    st.markdown("""
    ### 📝 About
    This app uses a deep learning model to classify pet breeds from images.
    Upload a clear photo for best results!
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Upload a clear photo of your pet",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Store image in session state
        st.session_state.uploaded_image = Image.open(uploaded_file)
        
        # Display the uploaded image at a fixed width
        st.image(
            st.session_state.uploaded_image, 
            caption="Your pet's photo",
            width=400
        )
        
        # Predict button
        if st.button("🔍 Identify Breed", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing your pet... Please wait"):
                try:
                    # Prepare the image for API call
                    img_bytes = BytesIO()
                    st.session_state.uploaded_image.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    
                    # Make API call
                    files = {'file': ('image.png', img_bytes, 'image/png')}
                    response = requests.post(f"{API_URL}?topk={topk}", files=files, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.prediction_result = result
                        st.rerun()  # Rerun to show results
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                        st.write(response.text)
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to the API. Make sure it's running on http://localhost:8000")
                except requests.exceptions.Timeout:
                    st.error("❌ Request timed out. Please try again.")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

with col2:
    st.subheader("🎯 Results")
    
    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result
        
        pred = result.get("prediction", {})
        label = pred.get("label", "Unknown")
        conf = pred.get("confidence", 0.0)
        conf_percent = conf * 100
        
        # Main result card
        st.markdown(f"""
        <div class="result-card">
            <div class="breed-name">{label.title()}</div>
            <div class="confidence-box">
                <strong>Confidence:</strong> {conf_percent:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar for confidence
        st.progress(conf, text=f"Confidence: {conf_percent:.1f}%")
        
        # Metrics
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Breed", label.title())
        with metric_col2:
            st.metric("Confidence", f"{conf_percent:.1f}%")
        
        # Top predictions
        top_k = result.get("top_k", [])
        if top_k:
            st.markdown("### 📊 Top Predictions")
            
            for i, item in enumerate(top_k, start=1):
                item_label = item.get('label', 'Unknown')
                item_conf = item.get('confidence', 0.0) * 100
                is_top = i == 1
                
                # Create prediction item
                with st.container():
                    item_col1, item_col2, item_col3 = st.columns([0.1, 0.6, 0.3])
                    
                    with item_col1:
                        if is_top:
                            st.markdown("### 🥇")
                        else:
                            st.markdown(f"### {i}.")
                    
                    with item_col2:
                        st.markdown(f"**{item_label.title()}**")
                    
                    with item_col3:
                        st.markdown(f"**{item_conf:.1f}%**")
                        st.progress(item_conf / 100)
                    
                    st.markdown("---")
    else:
        st.info("👆 Upload an image and click 'Identify Breed' to see results here")

# Instructions and tips section
st.markdown("---")

with st.expander("📖 Instructions & Tips", expanded=False):
    col_inst, col_tips = st.columns(2)
    
    with col_inst:
        st.markdown("""
        ### 📝 How to Use
        1. **Upload** a clear photo of your pet (JPG, JPEG, or PNG)
        2. **Click** "Identify Breed" to analyze the image
        3. **View** the predicted breed and confidence level
        4. **Check** the top predictions to see other possibilities
        """)
    
    with col_tips:
        st.markdown("""
        ### 💡 Tips for Better Results
        - ✅ Use clear, well-lit photos
        - ✅ Make sure your pet is the main subject
        - ✅ Use photos with good contrast
        - ✅ Try different angles if confidence is low
        - ✅ Avoid photos with multiple animals
        - ❌ Avoid blurry or dark images
        """)

# API Status Check
with st.expander("🔌 API Status", expanded=False):
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success("✅ API is running and ready!")
            st.json(health_data)
        else:
            st.warning(f"⚠️ API returned status {health_response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure it's running on http://localhost:8000")
    except Exception as e:
        st.error(f"❌ Error checking API status: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem 0;'>
    <p>Made with ❤️ for pet lovers | Powered by FastAI</p>
</div>
""", unsafe_allow_html=True)
