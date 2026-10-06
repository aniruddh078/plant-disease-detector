import os
import time
from datetime import datetime
import numpy as np
import streamlit as st
from PIL import Image
from predict import get_predictor, CLASS_NAMES

# Page configuration
st.set_page_config(
    page_title="AgriVision · Plant Health AI",
    page_icon="🌿",
    layout="centered",  # Centered layout looks 10x cleaner & native on phones!
    initial_sidebar_state="collapsed"
)

# Apple & Stripe-Grade Minimalist Modern CSS (Light & High-Contrast)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* 1. HIDE ALL STREAMLIT BRANDING, RED CROWN & PROFILE BADGE (THE CIRCLED ITEMS) */
    #MainMenu,
    header[data-testid="stHeader"],
    footer,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="manage-app-button"],
    [class*="viewerBadge"],
    [class*="manage-app"],
    .viewerBadge_container__r5tak,
    .viewerBadge_link__qRIco,
    iframe[title="streamlit_app"],
    button[title="View app in Streamlit Community Cloud"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        pointer-events: none !important;
    }

    /* 2. BASE APP CANVAS & TYPOGRAPHY */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        -webkit-font-smoothing: antialiased;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
        max-width: 680px !important; /* Perfect mobile & tablet reading width */
        margin: 0 auto;
    }

    /* 3. SLEEK CORPORATE NAVBAR / HEADER */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1.2rem;
        margin-bottom: 1.4rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .brand-title {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #ECFDF5;
        color: #059669;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        border: 1px solid #A7F3D0;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        background: #10B981;
        border-radius: 50%;
    }

    /* 4. INTRO HERO */
    .intro-section {
        margin-bottom: 1.5rem;
    }
    .intro-title {
        font-size: clamp(1.7rem, 5vw, 2.2rem);
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
        letter-spacing: -0.6px;
        margin: 0 0 0.4rem 0;
    }
    .intro-desc {
        font-size: 0.95rem;
        color: #64748B;
        line-height: 1.5;
        margin: 0;
    }

    /* 5. BIG TECH CARD CONTAINER */
    .clean-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 1.4rem;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
        margin-bottom: 1.2rem;
    }

    /* 6. DIAGNOSTIC RESULT BADGES & CARDS */
    .result-badge-healthy {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #047857;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.6rem;
    }
    .result-badge-orange {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #B45309;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.6rem;
    }
    .result-badge-red {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        color: #B91C1C;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.6rem;
    }
    .result-badge-gray {
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        color: #475569;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.6rem;
    }

    .diagnosis-name {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.4px;
        margin: 0 0 0.3rem 0;
        line-height: 1.2;
    }

    /* 7. PRESCRIPTION ACTION BOX */
    .prescription-box {
        background: #F8FAFC;
        border-left: 4px solid #10B981;
        padding: 1rem 1.1rem;
        border-radius: 0 14px 14px 0;
        margin-top: 1rem;
        font-size: 0.92rem;
        color: #1E293B;
        line-height: 1.5;
    }

    /* 8. TOUCH-FRIENDLY BUTTONS */
    div.stButton > button {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        border: none !important;
        width: 100% !important;
        transition: transform 0.15s ease, opacity 0.15s ease !important;
    }
    div.stButton > button:hover {
        opacity: 0.92 !important;
        transform: translateY(-1px) !important;
    }

    /* Download button */
    div.stDownloadButton > button {
        background-color: #059669 !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.2rem !important;
        border: none !important;
        width: 100% !important;
    }

    /* Clean file uploader */
    [data-testid="stFileUploader"] {
        border-radius: 14px;
    }

    /* Footer */
    .clean-footer {
        text-align: center;
        padding-top: 2rem;
        color: #94A3B8;
        font-size: 0.82rem;
        border-top: 1px solid #E2E8F0;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cached_predictor():
    return get_predictor()


def generate_prescription_text(result, area_sqm=10):
    now_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
    report = f"""======================================================
🌿 AGRIVISION · DIGITAL PLANT HEALTH PRESCRIPTION
======================================================
Issued On   : {now_str}
Project     : B.Tech Major Project by Aniruddh (MR078)

[1] CLINICAL FINDINGS
------------------------------------------------------
Diagnosis   : {result['title']}
Certainty   : {result['confidence']:.2f}%
Status      : {result['class_key'].upper()}

[2] SYMPTOMS
------------------------------------------------------
{result['symptoms']}

[3] TREATMENT PLAN & FERTILIZER
------------------------------------------------------
{result['remedy']}

[4] ESTIMATED FIELD APPLICATION (~{area_sqm} sq. meters)
------------------------------------------------------
"""
    if result['class_key'] == 'nitrogen':
        report += f"• Urea (46% Nitrogen) : {area_sqm * 15} grams\n"
        report += f"• Organic Compost      : {area_sqm * 0.4:.1f} kg\n"
        report += f"• Recommended Water    : {max(5, int(area_sqm * 1.5))} Liters\n"
    elif result['class_key'] == 'potassium':
        report += f"• MOP (Potash 60% K)   : {area_sqm * 12} grams\n"
        report += f"• Wood Ash Alternative : {area_sqm * 25} grams\n"
    else:
        report += "• Foliage is healthy. Standard maintenance recommended.\n"

    report += """
======================================================
Generated by AgriVision Deep Neural Network
======================================================
"""
    return report


def main():
    # 1. Clean Top Brand Header
    st.markdown("""
    <div class="top-nav">
        <div class="brand-title">
            <span>🌿</span> AgriVision
        </div>
        <div class="status-pill">
            <span class="status-dot"></span> AI Active
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Hero Introduction
    st.markdown("""
    <div class="intro-section">
        <h1 class="intro-title">Plant Health & Deficiency Scanner</h1>
        <p class="intro-desc">
            Instantly detect nutrient deficiencies and diseases from leaf photos with clinical accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar (Clean and uncluttered)
    with st.sidebar:
        st.markdown("### 🌿 About AgriVision")
        st.write("""
        AgriVision is an autonomous diagnostic tool built with Deep Convolutional Neural Networks 
        to detect crop nutrient stress before permanent yield loss occurs.
        """)

        st.markdown("#### 🔬 Target Deficiencies")
        st.markdown("""
        - 🟢 **Healthy Leaf:** Balanced nutrition
        - 🟠 **Nitrogen (N):** Yellowing from leaf tips
        - 🔴 **Potassium (K):** Leaf margin edge scorching
        - ⚪ **Unknown:** Inconclusive pattern
        """)

        st.markdown("---")
        st.markdown("#### 🎓 Project Credentials")
        st.caption("**Lead Developer:** Aniruddh (MR078)")
        st.caption("**Degree:** B.Tech Computer Science")
        st.caption("**Model Specs:** Edge TFLite CNN (10.6 MB)")

    # Load Model
    try:
        predictor = load_cached_predictor()
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        return

    # 3. Clean Card: Input Selection
    st.markdown("""
    <div class="clean-card">
        <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.6rem; color: #0F172A;">
            Select Photo Source
        </div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Choose option:",
        ["📁 Upload from Gallery", "📷 Use Camera", "🧪 Try Sample Image"],
        horizontal=True,
        label_visibility="collapsed"
    )

    image_to_predict = None

    if input_mode == "📁 Upload from Gallery":
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            image_to_predict = Image.open(uploaded_file)

    elif input_mode == "📷 Use Camera":
        camera_file = st.camera_input("Take a photo", label_visibility="collapsed")
        if camera_file is not None:
            image_to_predict = Image.open(camera_file)

    elif input_mode == "🧪 Try Sample Image":
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
                    st.image(thumb, use_container_width=True)
                    if st.button(f"Sample {idx+1}", key=f"sample_btn_{idx}"):
                        st.session_state["active_sample"] = img_path

            if "active_sample" in st.session_state:
                image_to_predict = Image.open(st.session_state["active_sample"])
        else:
            st.info("Use the upload or camera option above to scan a leaf.")

    st.markdown("</div>", unsafe_allow_html=True)

    # 4. Diagnostic Card (When Image is Ready)
    if image_to_predict is not None:
        start_time = time.time()
        with st.spinner("Analyzing leaf with neural model..."):
            result = predictor.predict(image_to_predict)
        latency_ms = (time.time() - start_time) * 1000

        # Balloons for healthy leaf!
        if result['class_key'] == 'healthy':
            st.balloons()

        badge_class = f"result-badge-{result['badge_color']}"

        st.markdown(f"""
        <div class="clean-card">
            <span class="{badge_class}">Diagnostic Result</span>
            <h2 class="diagnosis-name">{result['title']}</h2>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 0.4rem; padding-bottom: 0.8rem; border-bottom: 1px solid #F1F5F9;">
                <span style="font-weight: 800; font-size: 1.1rem; color: #059669;">
                    {result['confidence']:.1f}% Confidence
                </span>
                <span style="font-size: 0.8rem; color: #64748B; background: #F8FAFC; padding: 0.2rem 0.6rem; border-radius: 6px; border: 1px solid #E2E8F0;">
                    {latency_ms:.0f} ms
                </span>
            </div>
            
            <div style="margin-top: 1rem;">
                <div style="font-weight: 700; font-size: 0.85rem; color: #475569; text-transform: uppercase; letter-spacing: 0.5px;">
                    Symptoms
                </div>
                <div style="color: #334155; font-size: 0.95rem; line-height: 1.5; margin-top: 0.2rem;">
                    {result['symptoms']}
                </div>
            </div>

            <div class="prescription-box">
                <div style="font-weight: 800; color: #065F46; font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.3rem;">
                    💊 Recommended Action
                </div>
                {result['remedy']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 5. Interactive Treatment Dosage & Export in Clean Expanders
        with st.expander("🧮 Calculate Exact Fertilizer Dosage"):
            area_val = st.slider("Garden / Field Area (sq. meters):", 1, 50, 10, 1)
            if result['class_key'] == 'nitrogen':
                st.write(f"• **Urea (46% N):** `{area_val * 15} grams` in `{max(5, int(area_val * 1.5))}L` water.")
                st.write(f"• **Organic Compost:** `{area_val * 0.4:.1f} kg`.")
            elif result['class_key'] == 'potassium':
                st.write(f"• **MOP (60% K₂O):** `{area_val * 12} grams` around drip line.")
                st.write(f"• **Wood Ash:** `{area_val * 25} grams` raked into topsoil.")
            elif result['class_key'] == 'healthy':
                st.write(f"🌱 Plant is healthy! For {area_val} m², maintain routine balanced irrigation.")

        with st.expander("📊 View Probability Breakdown"):
            for cls_name, prob in result['probabilities'].items():
                c_lbl, c_bar, c_val = st.columns([2, 3, 1])
                with c_lbl:
                    st.write(f"**{cls_name.capitalize()}**")
                with c_bar:
                    st.progress(min(prob / 100.0, 1.0))
                with c_val:
                    st.write(f"`{prob:.1f}%`")

        # Download Prescription Button
        prescription_data = generate_prescription_text(result, area_val if 'area_val' in locals() else 10)
        st.download_button(
            label="📥 Download Official Prescription (.txt)",
            data=prescription_data,
            file_name=f"agrivision_prescription_{result['class_key']}.txt",
            mime="text/plain"
        )

    # Clean Professional Footer
    st.markdown("""
    <div class="clean-footer">
        🌿 <b>AgriVision</b> · B.Tech Engineering Major Project by Aniruddh · Powered by Deep Learning
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
