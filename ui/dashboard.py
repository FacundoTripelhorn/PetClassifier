import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API_URL = "http://localhost:8000/predict"

# Page configuration
st.set_page_config(
    page_title="🐾 Pet Breed Classifier", 
    page_icon="🐕", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    try:
        with open('ui/styles.css', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        st.error("CSS file not found. Please ensure ui/styles.css exists.")
        return ""
    except Exception as e:
        st.error(f"Error loading CSS: {e}")
        return ""

css_content = load_css()
if css_content:
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🐾 Pet Breed Classifier 🐾</h1>
    <p>Discover your furry friend's breed with our AI-powered classifier!</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎯 How it works:")
    st.markdown("""
    1. 📸 Upload a clear photo of your pet
    2. 🤖 Our AI analyzes the image
    3. 🎉 Get instant breed identification!
    """)

st.markdown("""
<div class="upload-section">
    <h2 style="text-align: center; color: #2c3e50; margin-bottom: 1rem;">
        📸 Upload Your Pet's Photo
    </h2>
    <p style="text-align: center; color: #34495e; font-size: 1.1rem;">
        Choose a clear, well-lit photo of your cat or dog for the best results!
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo of your pet for breed identification"
)

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="🐾 Your Pet's Photo", use_column_width=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        classify_clicked = st.button('🔍 Identify Breed!', use_container_width=True)
    
    if classify_clicked:
        with st.spinner('🔍 Analyzing your pet\'s features...'):
            img_bytes = BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            files = {'file': ('image.png', img_bytes, 'image/png')}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            st.markdown(f"""
            <div class="result-card">
                <h2 style="color: #2c3e50; text-align: center; margin-bottom: 1rem;">
                    🎉 Breed Identified!
                </h2>
                <div style="text-align: center;">
                    <h3 style="color: #e74c3c; font-size: 2rem; margin: 0;">
                        {result['label']}
                    </h3>
                    <p style="color: #34495e; font-size: 1.2rem; margin: 0.5rem 0;">
                        Confidence: {result['probability']*100:.1f}%
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            confidence = result['probability'] * 100
            st.markdown(f"""
            <div class="confidence-visualization">
                <p class="confidence-label">Confidence Level:</p>
                <div class="confidence-bar-container">
                    <div class="confidence-bar-fill" style="width: {confidence}%;"></div>
                </div>
                <p class="confidence-percentage">{confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Detailed Analysis")
            
            tab1, tab2, tab3, tab4 = st.tabs(["🏆 Top Predictions", "📈 Confidence Chart", "🎯 Prediction Breakdown", "📋 Detailed Stats"])
            
            with tab1:
                st.markdown("#### 🏆 Top Predictions")
                
                for i, (label, prob) in enumerate(zip(result['topk_labels'], result['topk_probs'])):
                    is_top = i == 0
                    emoji = "🥇" if is_top else "🥈" if i == 1 else "🥉" if i == 2 else "🔸"
                    color = "#e74c3c" if is_top else "#f39c12" if i == 1 else "#3498db" if i == 2 else "#95a5a6"
                    css_class = "prediction-item prediction-item-top" if is_top else "prediction-item prediction-item-other"
                    
                    st.markdown(f"""
                    <div class="{css_class}" style="border-left-color: {color};">
                        <div class="prediction-content">
                            <span class="prediction-label" style="color: {color};">
                                {emoji} {label}
                            </span>
                            <span class="prediction-confidence">
                                {prob*100:.1f}%
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab2:
                st.markdown("#### 📈 Confidence Distribution")
                
                df_predictions = pd.DataFrame({
                    'Breed': result['topk_labels'],
                    'Confidence': [p * 100 for p in result['topk_probs']]
                })
                
                colors = ['#e74c3c' if i == 0 else '#f39c12' if i == 1 else '#3498db' if i == 2 else '#95a5a6' 
                         for i in range(len(df_predictions))]
                
                fig = px.bar(
                    df_predictions, 
                    x='Confidence', 
                    y='Breed',
                    orientation='h',
                    color=colors,
                    title="Prediction Confidence by Breed",
                    labels={'Confidence': 'Confidence (%)', 'Breed': 'Pet Breed'},
                    color_discrete_map="identity"
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    height=max(300, len(df_predictions) * 50),
                    showlegend=False
                )
                
                fig.update_traces(
                    marker_line_width=0,
                    hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.markdown("#### 🎯 Prediction Breakdown")
                
                # Create a pie chart for top 5 predictions
                top_5_labels = result['topk_labels'][:5]
                top_5_probs = result['topk_probs'][:5]
                
                # Add "Others" if there are more than 5 predictions
                if len(result['topk_labels']) > 5:
                    others_prob = sum(result['topk_probs'][5:])
                    top_5_labels.append("Others")
                    top_5_probs.append(others_prob)
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=top_5_labels,
                    values=[p * 100 for p in top_5_probs],
                    hole=0.4,
                    marker_colors=['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6', '#95a5a6'][:len(top_5_labels)],
                    textinfo='label+percent',
                    textfont_size=12,
                    hovertemplate='<b>%{label}</b><br>Confidence: %{value:.1f}%<extra></extra>'
                )])
                
                fig_pie.update_layout(
                    title="Prediction Distribution",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    height=400,
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="middle",
                        y=0.5,
                        xanchor="left",
                        x=1.01
                    )
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with tab4:
                st.markdown("#### 📋 Detailed Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="🎯 Top Prediction",
                        value=f"{result['probability']*100:.1f}%",
                        delta=f"{result['probability']*100:.1f}% confidence"
                    )
                
                with col2:
                    confidence_gap = result['topk_probs'][0] - result['topk_probs'][1] if len(result['topk_probs']) > 1 else result['topk_probs'][0]
                    st.metric(
                        label="📊 Confidence Gap",
                        value=f"{confidence_gap*100:.1f}%",
                        delta="vs 2nd place"
                    )
                
                with col3:
                    st.metric(
                        label="🔢 Total Predictions",
                        value=len(result['topk_labels']),
                        delta="breeds analyzed"
                    )
                
                with col4:
                    avg_confidence = sum(result['topk_probs']) / len(result['topk_probs'])
                    st.metric(
                        label="📈 Avg Confidence",
                        value=f"{avg_confidence*100:.1f}%",
                        delta="across all predictions"
                    )
                
                st.markdown("#### 📋 Complete Prediction Results")
                
                df_detailed = pd.DataFrame({
                    'Rank': [f"#{i+1}" for i in range(len(result['topk_labels']))],
                    'Breed': result['topk_labels'],
                    'Confidence (%)': [f"{p*100:.2f}%" for p in result['topk_probs']],
                    'Confidence (Decimal)': [f"{p:.4f}" for p in result['topk_probs']],
                    'Status': ['🥇 Winner' if i == 0 else '🥈 Runner-up' if i == 1 else '🥉 Third' if i == 2 else '📊 Other' 
                              for i in range(len(result['topk_labels']))]
                })
                
                st.dataframe(
                    df_detailed,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Rank": st.column_config.TextColumn("Rank", width="small"),
                        "Breed": st.column_config.TextColumn("Breed Name", width="medium"),
                        "Confidence (%)": st.column_config.TextColumn("Confidence %", width="small"),
                        "Confidence (Decimal)": st.column_config.TextColumn("Raw Score", width="small"),
                        "Status": st.column_config.TextColumn("Status", width="small")
                    }
                )
            
        else:
            st.error(f"❌ Oops! Something went wrong: {response.text}")
            st.markdown("""
            <div class="error-tips">
                <strong>💡 Tips for better results:</strong><br>
                • Make sure the image is clear and well-lit<br>
                • Try to get a good view of your pet's face and body<br>
                • Ensure your pet is the main subject of the photo<br>
                • Supported formats: JPG, JPEG, PNG
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🐾 Made with ❤️ for pet lovers everywhere 🐾</p>
    <p>Upload another photo to discover more breeds!</p>
</div>
""", unsafe_allow_html=True)