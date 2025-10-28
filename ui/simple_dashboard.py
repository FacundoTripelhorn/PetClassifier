import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import json

# Page configuration
st.set_page_config(
    page_title="🐾 Pet Breed Classifier", 
    page_icon="🐕", 
    layout="centered"
)

# API configuration
API_URL = "http://localhost:8000/predict"

# Header
st.title("🐾 Pet Breed Classifier")
st.markdown("Upload a photo of your pet to identify its breed!")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear photo of your pet"
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Your pet's photo", use_column_width=True)
    
    # Predict button
    if st.button("🔍 Identify Breed", type="primary"):
        with st.spinner("Analyzing your pet..."):
            try:
                # Prepare the image for API call
                img_bytes = BytesIO()
                image.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                # Make API call
                files = {'file': ('image.png', img_bytes, 'image/png')}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result['prediction']

                    st.markdown(f"Prediction: {result}")
                    
                    # Display results
                    st.success("✅ Breed identified!")
                    
                    # Extract prediction details
                    if hasattr(prediction, '__len__') and len(prediction) >= 3:
                        # FastAI prediction format: (prediction, index, probabilities)
                        pred_label = str(prediction[0])
                        
                        st.markdown(f"""
                        ### 🎉 Result
                        **Breed:** {pred_label}  
                        """)
                        
                        # Show top 5 predictions if available
                        if hasattr(probabilities, 'topk'):
                            top_probs, top_indices = probabilities.topk(5)
                            st.markdown("### 📊 Top 5 Predictions:")
                            
                            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                                breed = str(prediction[0]) if idx == pred_idx else f"Breed #{idx}"
                                percentage = float(prob) * 100
                                st.write(f"{i+1}. {breed}: {percentage:.1f}%")
                    else:
                        # Fallback for different prediction formats
                        st.markdown(f"""
                        ### 🎉 Result
                        **Prediction:** {prediction}
                        """)
                        
                else:
                    st.error(f"❌ Error: {response.status_code}")
                    st.write(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the API. Make sure it's running on http://localhost:8000")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

# Instructions
st.markdown("---")
st.markdown("""
### 📝 Instructions
1. Upload a clear photo of your pet (JPG, JPEG, or PNG)
2. Click "Identify Breed" to analyze the image
3. View the predicted breed and confidence level

### 💡 Tips for better results
- Use clear, well-lit photos
- Make sure your pet is the main subject
- Try different angles if the first result isn't confident
""")

# Footer
st.markdown("---")
st.markdown("*Made with ❤️ for pet lovers*")
