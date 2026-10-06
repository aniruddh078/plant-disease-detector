import os
import streamlit as st
from PIL import Image
from predict import get_predictor, CLASS_NAMES

# Page configuration
st.set_page_config(
    page_title="Plant Nutrient & Health Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished card UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: #2E7D32;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .badge-healthy {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-orange {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-red {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-gray {
        background-color: #ECEFF1;
        color: #455A64;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cached_predictor():
    """Load model once into memory."""
    return get_predictor()


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌿 Plant Nutrient & Health Detector</h1>
        <p>AI-Powered Plant Health Diagnostics & Nutrient Deficiency Identification</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About the Project")
        st.write("""
        This web application uses a **Deep Convolutional Neural Network (CNN)** 
        trained on leaf imagery to diagnose nutrient shortages and plant health status.
        """)
        
        st.subheader("🎯 Supported Conditions")
        st.markdown("""
        - 🟢 **Healthy Leaf**
        - 🟠 **Nitrogen (N) Deficiency**
        - 🔴 **Potassium (K) Deficiency**
        - ⚪ **Inconclusive / Unknown**
        """)

        st.subheader("📊 Model Specifications")
        st.markdown("""
        - **Architecture:** 3-block Deep CNN
        - **Input Resolution:** 224 × 224 pixels
        - **Validation Accuracy:** ~89.1%
        - **Optimized for:** Mobile & Cloud Edge Inference
        """)

        st.markdown("---")
        st.caption("Developed as an AI in Agriculture B.Tech Project.")

    # Load Model
    try:
        predictor = load_cached_predictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Image Input Options
    st.write("### 📸 Provide a Plant Leaf Image")
    
    input_method = st.radio(
        "Choose how you want to provide an image:",
        ["📁 Upload from Device", "📷 Take Photo with Camera", "🧪 Use Sample Test Image"],
        horizontal=True
    )

    image_to_predict = None

    if input_method == "📁 Upload from Device":
        uploaded_file = st.file_uploader(
            "Choose a leaf image...", 
            type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded_file is not None:
            image_to_predict = Image.open(uploaded_file)

    elif input_method == "📷 Take Photo with Camera":
        camera_file = st.camera_input("Take a photo of the plant leaf:")
        if camera_file is not None:
            image_to_predict = Image.open(camera_file)

    elif input_method == "🧪 Use Sample Test Image":
        test_dir = os.path.join(os.path.dirname(__file__), "test_images")
        if os.path.exists(test_dir):
            sample_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            if sample_files:
                selected_sample = st.selectbox("Select a sample image from the test set:", sample_files)
                sample_path = os.path.join(test_dir, selected_sample)
                image_to_predict = Image.open(sample_path)
            else:
                st.warning("No test images found in `test_images/` folder.")
        else:
            st.warning("`test_images/` folder not found.")

    # Prediction Section
    if image_to_predict is not None:
        col_img, col_pred = st.columns([1, 1], gap="large")

        with col_img:
            st.subheader("🖼️ Leaf Image Preview")
            st.image(image_to_predict, use_container_width=True)

        with col_pred:
            st.subheader("🔬 AI Diagnostic Result")
            with st.spinner("Analyzing leaf patterns..."):
                result = predictor.predict(image_to_predict)

            badge_class = f"badge-{result['badge_color']}"
            st.markdown(f"""
            <div class="{badge_class}">
                <h3 style="margin:0; padding:0;">{result['title']}</h3>
            </div>
            """, unsafe_allow_html=True)

            st.write(f"**Confidence:** `{result['confidence']:.2f}%`")
            st.progress(min(result['confidence'] / 100.0, 1.0))

            # Details
            st.markdown("#### 🔍 Symptoms & Biological Analysis")
            st.write(result['symptoms'])

            # Recommendation Box
            st.markdown("#### 💡 Recommended Farmer Action & Treatment")
            st.info(result['remedy'])

            # Probability Breakdown
            with st.expander("📊 View All Class Confidence Probabilities"):
                for cls_name, prob in result['probabilities'].items():
                    col_label, col_val = st.columns([2, 1])
                    with col_label:
                        st.write(f"• **{cls_name.capitalize()}**")
                    with col_val:
                        st.write(f"`{prob:.2f}%`")
                    st.progress(min(prob / 100.0, 1.0))
    else:
        st.info("👆 Please upload an image, snap a photo, or select a sample image above to see the diagnosis.")


if __name__ == "__main__":
    main()
