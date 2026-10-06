# 🌿 Plant Nutrient Deficiency & Health Classifier

An end-to-end Machine Learning web application designed to identify plant health and nutrient deficiencies (Nitrogen and Potassium) from leaf images.

---

## 📌 Project Overview
- **Goal:** Early detection of plant nutritional stress to help farmers apply precise fertilizer treatments.
- **Model:** Deep Convolutional Neural Network (CNN) built with TensorFlow/Keras.
- **Input Image Size:** 224 × 224 pixels (RGB)
- **Validation Accuracy:** ~89.1%
- **Web Interface:** Streamlit (Mobile-friendly, Camera support, Instant diagnosis)

---

## 🎯 Classes Detected
1. **Healthy Leaf:** Normal green foliage with optimal nutrient levels.
2. **Nitrogen (N) Deficiency:** General chlorosis (yellowing) starting from older lower leaves.
3. **Potassium (K) Deficiency:** Marginal leaf necrosis and scorching ('leaf burn') on outer edges.
4. **Unknown / Inconclusive:** Unclear pattern or non-plant image.

---

## 📁 Project Structure
```
plant_project/
│
├── dataset/                  # Dataset with 1,335 images across 4 classes
├── test_images/              # Test sample images for verification
├── plant_model.keras         # Trained Keras CNN model weights
├── predict.py                # Standalone inference & diagnosis module
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
└── Untitled.ipynb            # Model training notebook
```

---

## 🚀 Running the App Locally
1. Activate your Python environment:
   ```bash
   C:\Users\Aniru\plant_env\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the web application:
   ```bash
   streamlit run app.py
   ```
4. Open the displayed local URL (`http://localhost:8501`) or Network URL on your phone.
