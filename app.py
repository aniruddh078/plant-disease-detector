import os
import time
from datetime import datetime
import streamlit as st
from PIL import Image
from predict import get_predictor, CLASS_NAMES

# Page configuration
st.set_page_config(
    page_title="AgriVision AI · Plant Health Diagnostic",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapsed by default so mobile users see main screen directly
)

# Polished Mobile-First & Desktop Adaptive CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Remove excessive top blank space on mobile and desktop */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 1000px !important;
        margin: 0 auto;
    }

    /* Prevent awkward horizontal scroll on phone */
    body {
        overflow-x: hidden !important;
    }

    /* Hero Banner - Mobile Responsive */
    .hero-container {
        background: linear-gradient(135deg, #0f3d24 0%, #1b5e34 60%, #2e7d47 100%);
        padding: 1.8rem 1.6rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(15, 61, 36, 0.15);
        text-align: left;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255, 255, 255, 0.16);
        backdrop-filter: blur(8px);
        padding: 0.3rem 0.75rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    .hero-title {
        font-size: clamp(1.6rem, 4vw, 2.3rem);
        font-weight: 800;
        margin: 0;
        line-height: 1.25;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: clamp(0.85rem, 2.5vw, 1rem);
        opacity: 0.92;
        margin-top: 0.4rem;
        margin-bottom: 0;
        line-height: 1.4;
    }

    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.6rem;
        margin-bottom: 1.4rem;
    }
    @media (max-width: 680px) {
        .stats-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.85rem 0.6rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .metric-val {
        font-size: clamp(1.2rem, 3vw, 1.45rem);
        font-weight: 800;
        color: #1b5e20;
        line-height: 1.1;
    }
    .metric-lbl {
        font-size: 0.72rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-top: 0.25rem;
    }

    /* Adaptive Diagnosis Result Card */
    .result-card {
        border-radius: 16px;
        padding: 1.4rem;
        margin-top: 0.8rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(0,0,0,0.06);
    }
    .card-healthy {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
        border-left: 6px solid #2e7d32;
    }
    .card-orange {
        background: linear-gradient(135deg, #fff3e0 0%, #fff8e1 100%);
        border-left: 6px solid #ef6c00;
    }
    .card-red {
        background: linear-gradient(135deg, #ffebee 0%, #fce4ec 100%);
        border-left: 6px solid #c62828;
    }
    .card-gray {
        background: linear-gradient(135deg, #eceff1 0%, #f5f5f5 100%);
        border-left: 6px solid #546e7a;
    }

    .result-header-tag {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        opacity: 0.8;
    }
    .result-title-text {
        font-size: clamp(1.3rem, 3.5vw, 1.7rem);
        font-weight: 800;
        color: #1e293b;
        margin: 0.3rem 0;
        line-height: 1.2;
    }

    /* Mobile Action Box */
    .action-box {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        margin-top: 0.8rem;
        font-size: 0.9rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
        color: #94a3b8;
        font-size: 0.8rem;
    }

    /* Touch-friendly buttons */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cached_predictor():
    return get_predictor()


def generate_report_text(result):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""=====================================================
🌿 AGRIVISION AI · PLANT HEALTH DIAGNOSTIC REPORT
=====================================================
Generated On: {now_str}

[1] DIAGNOSTIC SUMMARY
-----------------------------------------------------
Condition Detected : {result['title']}
Confidence Level   : {result['confidence']:.2f}%
Status Category    : {result['class_key'].upper()}

[2] SYMPTOMS & OBSERVATION
-----------------------------------------------------
{result['symptoms']}

[3] AGRONOMIST RECOMMENDATION & TREATMENT
-----------------------------------------------------
{result['remedy']}

[4] CONFIDENCE DISTRIBUTION
-----------------------------------------------------
"""
    for cls, prob in result['probabilities'].items():
        report += f"- {cls.capitalize():12} : {prob:.2f}%\n"

    report += """
=====================================================
Developed by Aniruddh (MR078) · B.Tech Project
AgriVision AI Diagnostic System
=====================================================
"""
    return report


def main():
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">
            <span>🟢</span> AI Engine Active · Edge Optimized
        </div>
        <h1 class="hero-title">🌿 AgriVision AI</h1>
        <p class="hero-subtitle">
            Smart Plant Health & Real-Time Nutrient Deficiency Detection
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Adaptive 4-Tile Stats Strip (Automatically adapts to 2x2 on phones)
    st.markdown("""
    <div class="stats-grid">
        <div class="metric-card">
            <div class="metric-val">89.1%</div>
            <div class="metric-lbl">Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">10.6 MB</div>
            <div class="metric-lbl">Model Size</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">&lt; 50 ms</div>
            <div class="metric-lbl">Speed</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">4 Types</div>
            <div class="metric-lbl">N, K & Health</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🌾 Plant Doctor Assistant")
        st.info("💡 **Best Results:** Take close-up photos of leaves in bright daylight against a plain background.")

        st.markdown("#### 🔬 Nutrient Deficiencies 101")
        with st.expander("🌱 Nitrogen (N) Role"):
            st.write("""
            **Function:** Critical for chlorophyll and stem/foliage growth.  
            **Symptoms:** Yellowing (chlorosis) starting on older lower leaves.
            """)
        with st.expander("🍂 Potassium (K) Role"):
            st.write("""
            **Function:** Regulates water retention and plant disease resistance.  
            **Symptoms:** Browning/scorching along leaf tips and outer edges.
            """)
        with st.expander("🌿 Healthy Leaves"):
            st.write("""
            **Characteristics:** Vibrant, uniform green pigmentation with balanced leaf turgidity.
            """)

        st.markdown("---")
        st.markdown("### 🎓 Project Details")
        st.caption("**Developer:** Aniruddh (MR078)")
        st.caption("**Domain:** Computer Vision & AI in Agriculture")
        st.caption("**Model:** Custom 3-Block Deep CNN")

    # Load Model Safely
    try:
        predictor = load_cached_predictor()
    except Exception as e:
        st.error(f"⚠️ Model initialization error: {e}")
        st.info("Ensure `plant_model.tflite` is uploaded in the repository.")
        return

    # Image Input Selection
    st.markdown("### 📸 Choose Input Method")
    tab_upload, tab_camera, tab_sample = st.tabs([
        "📁 Upload Photo", 
        "📷 Phone / Web Camera", 
        "🧪 Demo Samples"
    ])

    image_to_predict = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Select an image from your gallery:", 
            type=["jpg", "jpeg", "png", "webp"],
            help="High-contrast leaf photos produce highest diagnostic accuracy."
        )
        if uploaded_file is not None:
            image_to_predict = Image.open(uploaded_file)

    with tab_camera:
        camera_file = st.camera_input("Snap a live photo of the leaf:")
        if camera_file is not None:
            image_to_predict = Image.open(camera_file)

    with tab_sample:
        st.write("Click any sample image below for an instant demo:")
        test_dir = os.path.join(os.path.dirname(__file__), "test_images")
        sample_files = []
        if os.path.exists(test_dir):
            sample_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

        if sample_files:
            cols = st.columns(min(len(sample_files), 3))
            for idx, sfile in enumerate(sample_files[:3]):
                with cols[idx]:
                    img_path = os.path.join(test_dir, sfile)
                    thumb = Image.open(img_path)
                    st.image(thumb, caption=f"Sample {idx+1}", use_container_width=True)
                    if st.button(f"Scan Sample {idx+1}", key=f"btn_sample_{idx}", use_container_width=True):
                        st.session_state["selected_sample_img"] = img_path

            if "selected_sample_img" in st.session_state:
                image_to_predict = Image.open(st.session_state["selected_sample_img"])
        else:
            st.info("Upload or take a photo above to test your plant leaf!")

    # Output Section
    if image_to_predict is not None:
        st.write("---")
        st.markdown("## 🔍 Diagnostic Result")

        col_img, col_result = st.columns([1, 1.15], gap="medium")

        with col_img:
            st.markdown("#### 🖼️ Uploaded Leaf")
            st.image(image_to_predict, use_container_width=True)

        with col_result:
            start_time = time.time()
            with st.spinner("Analyzing neural features..."):
                result = predictor.predict(image_to_predict)
            calc_latency = (time.time() - start_time) * 1000

            # Balloons on Healthy plant!
            if result['class_key'] == 'healthy':
                st.balloons()
                st.toast("🎉 Plant leaf is healthy!", icon="🌿")
            else:
                st.toast(f"⚠️ {result['title']} detected!", icon="🔍")

            # Diagnosis Card
            card_class = f"card-{result['badge_color']}"
            st.markdown(f"""
            <div class="result-card {card_class}">
                <div class="result-header-tag">AI Prediction</div>
                <div class="result-title-text">{result['title']}</div>
                <div style="display:flex; align-items:center; gap:0.8rem; margin-top:0.4rem; flex-wrap:wrap;">
                    <span style="font-weight:700; font-size:1.05rem; color:#0f172a;">
                        Confidence: {result['confidence']:.2f}%
                    </span>
                    <span style="font-size:0.8rem; color:#64748b; background:white; padding:0.2rem 0.5rem; border-radius:6px; border:1px solid #e2e8f0;">
                        ⏱️ {calc_latency:.1f} ms
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Mobile-optimized tabbed analysis
            subtab1, subtab2, subtab3 = st.tabs([
                "💡 Action Plan", 
                "📊 Probabilities", 
                "📥 Prescription"
            ])

            with subtab1:
                st.markdown("##### 🔬 Symptoms Detected")
                st.write(result['symptoms'])

                st.markdown("##### 💊 Recommended Treatment & Fertilizer")
                st.success(result['remedy'])

                st.markdown("""
                <div class="action-box">
                    <b>🚜 Agronomist Tip:</b> Apply fertilizers in the early morning or evening when leaf stomata are open. Avoid direct stem contact.
                </div>
                """, unsafe_allow_html=True)

            with subtab2:
                st.markdown("##### Class Probabilities")
                for cls_name, prob in result['probabilities'].items():
                    b_col, v_col = st.columns([3, 1])
                    with b_col:
                        st.write(f"**{cls_name.capitalize()}**")
                        st.progress(min(prob / 100.0, 1.0))
                    with v_col:
                        st.write(f"`{prob:.2f}%`")

            with subtab3:
                st.markdown("##### Download Agronomy Report")
                st.write("Save or print this diagnosis for project presentation:")
                report_data = generate_report_text(result)
                st.download_button(
                    label="📄 Download Prescription (.txt)",
                    data=report_data,
                    file_name=f"agrivision_report_{result['class_key']}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    else:
        st.info("👆 **Ready to scan:** Upload a leaf picture, use your phone camera, or pick a sample above!")

    # Footer
    st.markdown("""
    <div class="footer">
        🌿 <b>AgriVision AI</b> · Final Year B.Tech Project by Aniruddh · Powered by Deep Learning & Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
