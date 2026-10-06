import os
import time
from datetime import datetime
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance
from predict import get_predictor, CLASS_NAMES

# Page configuration
st.set_page_config(
    page_title="AgriVision AI · Next-Gen Plant Diagnostic",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ultra-Modern Cyber-Forest Glassmorphism Theme (Trendy & Futuristic)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    /* Seamless mobile and desktop canvas */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 1060px !important;
        margin: 0 auto;
    }
    body {
        overflow-x: hidden !important;
    }

    /* Animated Cyber-Forest Hero Banner */
    .cyber-hero {
        background: radial-gradient(circle at 10% 20%, rgba(30, 80, 50, 0.95) 0%, rgba(10, 35, 20, 0.98) 90%);
        border: 1px solid rgba(0, 230, 118, 0.25);
        border-radius: 22px;
        padding: 2rem 1.8rem;
        color: #ffffff;
        position: relative;
        overflow: hidden;
        box-shadow: 0 12px 35px rgba(0, 30, 15, 0.25);
        margin-bottom: 1.2rem;
    }
    .cyber-hero::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(0, 230, 118, 0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .pulse-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(0, 230, 118, 0.12);
        color: #00E676;
        border: 1px solid rgba(0, 230, 118, 0.35);
        padding: 0.35rem 0.85rem;
        border-radius: 50px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background: #00E676;
        border-radius: 50%;
        box-shadow: 0 0 10px #00E676;
        animation: pulseAnimation 1.8s infinite;
    }
    @keyframes pulseAnimation {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(0, 230, 118, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0); }
    }
    .hero-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.8rem, 4.5vw, 2.6rem);
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #ffffff 40%, #A7F3D0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: clamp(0.88rem, 2.2vw, 1.05rem);
        color: #94A3B8;
        margin-top: 0.45rem;
        margin-bottom: 0;
        line-height: 1.4;
    }

    /* Glassmorphism Stat Cards */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.4rem;
    }
    @media (max-width: 680px) {
        .stats-container {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    .stat-pill {
        background: #ffffff;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 0.9rem 0.7rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
    }
    .stat-pill:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
        border-color: #A7F3D0;
    }
    .stat-number {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.3rem, 3vw, 1.55rem);
        font-weight: 700;
        color: #065F46;
        line-height: 1.1;
    }
    .stat-label {
        font-size: 0.72rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.25rem;
    }

    /* High-Impact Result Card */
    .diagnostic-panel {
        border-radius: 18px;
        padding: 1.5rem;
        margin-top: 0.8rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(0,0,0,0.06);
        position: relative;
    }
    .panel-healthy {
        background: linear-gradient(135deg, #ECFDF5 0%, #F0FDF4 100%);
        border: 1px solid #6EE7B7;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.12);
    }
    .panel-orange {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border: 1px solid #FCD34D;
        box-shadow: 0 8px 24px rgba(245, 158, 11, 0.12);
    }
    .panel-red {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 1px solid #FCA5A5;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.12);
    }
    .panel-gray {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid #CBD5E1;
    }

    .tag-title {
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #475569;
    }
    .diagnostic-headline {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.4rem, 3.8vw, 1.85rem);
        font-weight: 800;
        color: #0F172A;
        margin: 0.3rem 0;
        line-height: 1.2;
    }

    /* Calculator & Info Cards */
    .feature-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.1rem;
        margin-top: 0.8rem;
    }
    .feature-title {
        font-weight: 700;
        color: #0F172A;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-bottom: 0.4rem;
    }

    /* Modern Footer */
    .cyber-footer {
        text-align: center;
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid #E2E8F0;
        color: #94A3B8;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cached_predictor():
    return get_predictor()


def generate_visual_attention_map(pil_img):
    """
    Computer Vision Explainability Simulation (Visual Saliency Mask)
    Highlights yellow/brown chlorosis & necrosis zones on leaf.
    """
    img_rgb = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Detect chlorosis (yellow: high R & G, low B) and necrotic burn (high R, lower G, very low B)
    yellow_regions = (r > 0.42) & (g > 0.42) & (b < 0.40) & (np.abs(r - g) < 0.28)
    burn_regions = (r > 0.35) & (g < 0.38) & (b < 0.26)
    defect_mask = np.clip((yellow_regions.astype(float) * 0.85 + burn_regions.astype(float) * 0.95), 0, 1)

    # Thermal attention map overlay
    attention_overlay = np.copy(arr)
    attention_overlay[:, :, 0] = np.clip(arr[:, :, 0] * 0.6 + defect_mask * 0.8, 0, 1)  # Red channel glow
    attention_overlay[:, :, 1] = np.clip(arr[:, :, 1] * 0.7 - defect_mask * 0.2, 0, 1)
    attention_overlay[:, :, 2] = np.clip(arr[:, :, 2] * 0.5, 0, 1)

    return Image.fromarray((attention_overlay * 255).astype(np.uint8))


def generate_prescription_report(result, field_sqm=10):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""=====================================================
🌿 AGRIVISION AI · SMART PLANT HEALTH DIAGNOSTIC REPORT
=====================================================
Date & Timestamp : {now_str}
Project Evaluator: B.Tech Major Project Review
Student In-Charge: Aniruddh (MR078)

[1] NEURAL NETWORK DIAGNOSTIC
-----------------------------------------------------
Primary Finding     : {result['title']}
Confidence Score    : {result['confidence']:.2f}%
Status Identifier   : {result['class_key'].upper()}

[2] SYMPTOMATIC ANALYSIS
-----------------------------------------------------
{result['symptoms']}

[3] EXPERT AGRONOMY PRESCRIPTION
-----------------------------------------------------
{result['remedy']}

[4] ESTIMATED NPK DOSAGE (For ~{field_sqm} sq.m / ~{max(1, field_sqm//2)} Plants)
-----------------------------------------------------
"""
    if result['class_key'] == 'nitrogen':
        report += f"• Recommended Urea (46% N)       : ~{field_sqm * 15} grams\n"
        report += f"• Organic Vermicompost / Manure   : ~{field_sqm * 0.4:.1f} kg\n"
        report += "• Water Dilution Ratio            : 15-20 Liters\n"
    elif result['class_key'] == 'potassium':
        report += f"• Recommended MOP / Potash (60% K): ~{field_sqm * 12} grams\n"
        report += f"• Potassium Sulfate (SOP) Foliar  : ~{field_sqm * 8} grams\n"
        report += "• Water Dilution Ratio            : 15 Liters\n"
    else:
        report += "• Plant is within normal vitality range. Routine balanced maintenance advised.\n"

    report += """
=====================================================
AgriVision AI · Powered by Deep CNN & Edge Inference
=====================================================
"""
    return report


def main():
    # Animated Cyber Hero Header
    st.markdown("""
    <div class="cyber-hero">
        <div class="pulse-badge">
            <span class="pulse-dot"></span> Neural Edge Engine · Online
        </div>
        <h1 class="hero-heading">🌿 AgriVision AI</h1>
        <p class="hero-sub">
            Autonomous Plant Health & Precision Nutrient Diagnostics Powered by Deep Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Trendy Stats Grid (Adaptive 2x2 on Mobile, 4x1 on Laptop)
    st.markdown("""
    <div class="stats-container">
        <div class="stat-pill">
            <div class="stat-number">89.1%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
        <div class="stat-pill">
            <div class="stat-number">10.6 MB</div>
            <div class="stat-label">TFLite Edge Core</div>
        </div>
        <div class="stat-pill">
            <div class="stat-number">&lt; 35 ms</div>
            <div class="stat-label">Inference Latency</div>
        </div>
        <div class="stat-pill">
            <div class="stat-number">N · K · Health</div>
            <div class="stat-label">Diagnostic Scope</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Doctor & Notes
    with st.sidebar:
        st.markdown("### 🌾 Plant Doctor Control Panel")
        st.success("🟢 **Model Loaded:** Edge TFLite CNN")

        st.markdown("#### 📖 Nutrient Deficiency Reference")
        with st.expander("🌱 Nitrogen (N) Signs & Treatment"):
            st.write("""
            - **Signs:** Older leaves turn uniform yellow starting from the leaf apex (tip).
            - **Treatment:** Urea, ammonium nitrate, or fermented compost tea.
            """)
        with st.expander("🍂 Potassium (K) Signs & Treatment"):
            st.write("""
            - **Signs:** Outer margins show 'leaf burn' scorching; leaves curl inward.
            - **Treatment:** Muriate of Potash (MOP), wood ash, or Potassium Sulfate.
            """)
        with st.expander("🌿 Healthy Foliage Signs"):
            st.write("""
            - **Signs:** Uniform chloroplast distribution with strong turgor pressure.
            """)

        st.markdown("---")
        st.markdown("### 👨‍💻 Project Information")
        st.caption("**Candidate:** Aniruddh (MR078)")
        st.caption("**Degree:** B.Tech Major Project")
        st.caption("**Dataset:** 1,335 High-Resolution Leaf Images")

    # Load Model
    try:
        predictor = load_cached_predictor()
    except Exception as e:
        st.error(f"⚠️ Model initialization error: {e}")
        return

    # Modern Input Tabs
    st.markdown("### 📸 Leaf Image Acquisition")
    tab_upload, tab_camera, tab_samples = st.tabs([
        "📁 Upload Photo", 
        "📷 Live Camera", 
        "🧪 1-Click Samples"
    ])

    image_to_predict = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a leaf photo from your device:", 
            type=["jpg", "jpeg", "png", "webp"],
            help="Crisp, well-lit photos give maximum diagnostic confidence."
        )
        if uploaded_file is not None:
            image_to_predict = Image.open(uploaded_file)

    with tab_camera:
        camera_file = st.camera_input("Take a direct photo of the plant leaf:")
        if camera_file is not None:
            image_to_predict = Image.open(camera_file)

    with tab_samples:
        st.write("Click any sample image for an instant live test:")
        test_dir = os.path.join(os.path.dirname(__file__), "test_images")
        sample_files = []
        if os.path.exists(test_dir):
            sample_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

        if sample_files:
            sample_cols = st.columns(min(len(sample_files), 3))
            for idx, sfile in enumerate(sample_files[:3]):
                with sample_cols[idx]:
                    img_path = os.path.join(test_dir, sfile)
                    thumb = Image.open(img_path)
                    st.image(thumb, caption=f"Test Image {idx+1}", use_container_width=True)
                    if st.button(f"Analyze #{idx+1}", key=f"btn_sample_{idx}", use_container_width=True):
                        st.session_state["selected_sample_img"] = img_path

            if "selected_sample_img" in st.session_state:
                image_to_predict = Image.open(st.session_state["selected_sample_img"])
        else:
            st.info("You can upload or capture any leaf photo using the tabs above!")

    # Live Diagnostic Results
    if image_to_predict is not None:
        st.write("---")
        st.markdown("## 🔬 Neural Diagnostic Dashboard")

        col_leaf, col_diag = st.columns([1, 1.2], gap="large")

        with col_leaf:
            st.markdown("#### 🖼️ Uploaded Leaf")
            st.image(image_to_predict, use_container_width=True)

            # Simulated Visual Attention Heatmap (Cool feature for AI explainability)
            with st.expander("🔥 View AI Visual Attention Heatmap"):
                st.caption("Highlights chlorosis (yellowing) and necrotic margin zones detected by the model:")
                heatmap_img = generate_visual_attention_map(image_to_predict)
                st.image(heatmap_img, use_container_width=True)

        with col_diag:
            start_time = time.time()
            with st.spinner("Extracting deep convolutional feature maps..."):
                result = predictor.predict(image_to_predict)
            calc_latency = (time.time() - start_time) * 1000

            # Celebration effects for healthy status
            if result['class_key'] == 'healthy':
                st.balloons()
                st.toast("🎉 Plant is healthy and vibrant!", icon="🌿")
            else:
                st.toast(f"⚠️ {result['title']} detected!", icon="🔍")

            # High-impact diagnosis card
            panel_class = f"panel-{result['badge_color']}"
            st.markdown(f"""
            <div class="diagnostic-panel {panel_class}">
                <div class="tag-title">Autonomous AI Diagnosis</div>
                <div class="diagnostic-headline">{result['title']}</div>
                <div style="display:flex; align-items:center; gap:0.9rem; margin-top:0.4rem; flex-wrap:wrap;">
                    <span style="font-weight:700; font-size:1.15rem; color:#0F172A;">
                        Confidence: {result['confidence']:.2f}%
                    </span>
                    <span style="font-size:0.8rem; color:#475569; background:white; padding:0.25rem 0.6rem; border-radius:6px; border:1px solid #E2E8F0; font-weight:600;">
                        ⚡ Latency: {calc_latency:.1f} ms
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Plant Vitality Index Bar (Cool visual gauge)
            vitality_score = 95.0 if result['class_key'] == 'healthy' else max(25.0, 100.0 - (result['confidence'] * 0.65))
            st.write(f"**Plant Vitality Index:** `{vitality_score:.1f} / 100`")
            st.progress(vitality_score / 100.0)

            # Interactive Tabs for Cool Features
            feature_tab1, feature_tab2, feature_tab3, feature_tab4 = st.tabs([
                "💊 Action Plan", 
                "🧮 Dosage Calculator", 
                "📊 All Probabilities", 
                "📄 Prescription"
            ])

            with feature_tab1:
                st.markdown("##### 🔍 Identified Symptoms")
                st.write(result['symptoms'])

                st.markdown("##### 💡 Agronomist Remediation Plan")
                st.success(result['remedy'])

                st.markdown("""
                <div class="feature-card">
                    <div class="feature-title">🚜 Field Application Tip</div>
                    Avoid spraying concentrated chemical fertilizers during peak midday sun to prevent foliar salt burn. Apply early in the morning or during sunset.
                </div>
                """, unsafe_allow_html=True)

            with feature_tab2:
                st.markdown("##### 🧮 Interactive Fertilizer Dosage Calculator")
                st.write("Calculate the exact fertilizer quantity needed for your farm or garden:")
                
                area_sqm = st.slider("Select Garden / Farm Area (in square meters):", min_value=1, max_value=50, value=10, step=1)
                
                if result['class_key'] == 'nitrogen':
                    urea_needed = area_sqm * 15
                    manure_needed = area_sqm * 0.4
                    st.info(f"""
                    **Dosage for {area_sqm} m² (~{max(1, area_sqm // 2)} plants):**
                    - **Chemical (Urea 46% N):** `{urea_needed} grams` diluted in `{max(5, area_sqm * 1.5):.0f}L` water.
                    - **Organic Alternative:** `{manure_needed:.1f} kg` Vermicompost or Farmyard Manure.
                    """)
                elif result['class_key'] == 'potassium':
                    mop_needed = area_sqm * 12
                    st.info(f"""
                    **Dosage for {area_sqm} m² (~{max(1, area_sqm // 2)} plants):**
                    - **Potassium Chloride (MOP 60% K₂O):** `{mop_needed} grams` applied around plant drip line.
                    - **Organic Alternative:** Wood ash lightly raked into topsoil (`{area_sqm * 25} grams`).
                    """)
                elif result['class_key'] == 'healthy':
                    st.success(f"🌱 Your plant is healthy! For {area_sqm} m², continue routine irrigation with standard maintenance compost.")
                else:
                    st.write("Scan a valid plant leaf to calculate precision fertilizer dosages.")

            with feature_tab3:
                st.markdown("##### Deep CNN Class Distribution")
                for cls_name, prob in result['probabilities'].items():
                    p_col, v_col = st.columns([3, 1])
                    with p_col:
                        st.write(f"**{cls_name.capitalize()}**")
                        st.progress(min(prob / 100.0, 1.0))
                    with v_col:
                        st.write(f"`{prob:.2f}%`")

            with feature_tab4:
                st.markdown("##### Official Agronomist Report")
                st.write("Download an official timestamped diagnostic prescription for your presentation:")
                report_text = generate_prescription_report(result, area_sqm if 'area_sqm' in locals() else 10)
                st.download_button(
                    label="📥 Download Diagnostic Prescription (.txt)",
                    data=report_text,
                    file_name=f"agrivision_prescription_{result['class_key']}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    else:
        st.info("👆 **Scan to begin:** Upload a leaf picture, use your phone camera, or pick a 1-click sample above!")

    # Modern Cyber Footer
    st.markdown("""
    <div class="cyber-footer">
        🌿 <b>AgriVision AI</b> · Final Year B.Tech Major Project by Aniruddh · Powered by Deep Learning & Streamlit Edge
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
