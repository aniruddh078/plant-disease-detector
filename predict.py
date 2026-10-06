import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Suppress unnecessary TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

CLASS_NAMES = ["healthy", "nitrogen", "potassium", "unknown"]

CLASS_DETAILS = {
    "healthy": {
        "title": "Healthy Plant Leaf",
        "badge_color": "healthy",
        "description": "The leaf shows uniform green pigmentation, balanced nutrient reserves, and healthy cell structure.",
        "symptoms": "Uniform leaf coloration, no chlorosis (yellowing) or necrosis (browning), active foliage vigor.",
        "remedy": "Continue regular balanced irrigation, adequate sunlight, and standard routine care. No corrective fertilizer needed."
    },
    "nitrogen": {
        "title": "Nitrogen (N) Deficiency",
        "badge_color": "orange",
        "description": "Nitrogen is critical for chlorophyll formation and vegetative foliage expansion.",
        "symptoms": "General yellowing (chlorosis) beginning at the tip of older lower leaves, moving inward. Overall plant growth may be stunted.",
        "remedy": "Apply a fast-acting nitrogen fertilizer such as Urea, Calcium Ammonium Nitrate (CAN), or top-dress with aged organic compost / farmyard manure."
    },
    "potassium": {
        "title": "Potassium (K) Deficiency",
        "badge_color": "red",
        "description": "Potassium controls water regulation, stomatal opening, and plant immune defense.",
        "symptoms": "Marginal chlorosis and browning/scorching along leaf tips and outer edges ('leaf burn'), leaves curling upwards, brittle stems.",
        "remedy": "Apply potassium-rich fertilizer such as Muriate of Potash (MOP / Potassium Chloride), Potassium Sulfate (SOP), or well-balanced NPK (e.g. 10-10-20)."
    },
    "unknown": {
        "title": "Inconclusive / Unknown Pattern",
        "badge_color": "gray",
        "description": "The visual patterns do not match standard healthy or single-deficiency symptoms.",
        "symptoms": "Blurry photo, improper lighting, complex multi-stress symptoms, or non-leaf subject.",
        "remedy": "Please capture a crisp close-up photo of a single leaf under bright daylight against a plain background and scan again."
    }
}


def validate_leaf_image(image_input):
    """
    Biological Foliage & Computer Vision Guardrail:
    Determines whether the input image is a genuine plant leaf.
    Rejects human faces, selfies, rooms, clothes, vehicles, blank papers, etc.
    """
    img = image_input.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Convert to HSV color space
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    s = np.zeros_like(max_c)
    mask = max_c > 0
    s[mask] = delta[mask] / max_c[mask]

    h = np.zeros_like(max_c)
    mask_r = (max_c == r) & (delta > 0)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    mask_g = (max_c == g) & (delta > 0)
    h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
    mask_b = (max_c == b) & (delta > 0)
    h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
    h = h * 60.0  # Degrees 0-360

    # 1. Biological Foliage Signatures:
    # Healthy green chlorophyll: 32° to 160°, S > 0.12, V > 0.10
    green_mask = (h >= 32) & (h <= 160) & (s >= 0.12) & (max_c >= 0.10)
    # Chlorotic yellow leaf: 18° to 45°, S > 0.18, V > 0.18
    yellow_mask = (h >= 18) & (h <= 45) & (s >= 0.18) & (max_c >= 0.18)
    # Necrotic potassium edge burn: 10° to 30°, S 0.15-0.80, V 0.15-0.70
    brown_mask = (h >= 10) & (h <= 30) & (s >= 0.15) & (s <= 0.80) & (max_c >= 0.15) & (max_c <= 0.70)

    # 2. Human Skin Tone Rejection (Prevent selfies, faces, hands from being classified as leaves):
    skin_mask = (r > 0.35) & (g > 0.20) & (b > 0.15) & (r > g) & (g > b) & ((h <= 25) | (h >= 335)) & (s <= 0.65)
    skin_ratio = float(np.mean(skin_mask))

    # Plant foliage pixels (excluding skin overlap)
    plant_mask = (green_mask | yellow_mask | brown_mask) & (~skin_mask)
    foliage_ratio = float(np.mean(plant_mask))

    # 3. Organic Venation / Texture Complexity (Laplacian variance):
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    laplacian = np.abs(
        gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1] - 4 * gray[1:-1, 1:-1]
    )
    texture_score = float(np.mean(laplacian) * 100.0)

    # Validation criteria:
    if skin_ratio >= 0.25:
        return False, "Human face, skin, or portrait detected. Please upload a plant leaf photo."
    
    if foliage_ratio < 0.20:
        return False, "No plant foliage detected (appears to be an indoor room, object, animal, or non-plant item)."
    
    if texture_score < 1.4:
        return False, "Flat or artificial graphic detected with no organic leaf texture."

    return True, "Valid leaf verified."


class PlantPredictor:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        tflite_path = os.path.join(base_dir, "plant_model.tflite")
        h5_path = os.path.join(base_dir, "plant_model_deploy.h5")
        keras_path = os.path.join(base_dir, "plant_model.keras")

        if os.path.exists(tflite_path):
            self.mode = "tflite"
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        elif os.path.exists(h5_path):
            self.mode = "keras"
            self.model = tf.keras.models.load_model(h5_path)
        elif os.path.exists(keras_path):
            self.mode = "keras"
            self.model = tf.keras.models.load_model(keras_path)
        else:
            raise FileNotFoundError("No model file found.")

    def predict(self, image_input):
        if not isinstance(image_input, Image.Image):
            image_input = Image.open(image_input)

        # Fix mobile phone orientation
        try:
            image_input = ImageOps.exif_transpose(image_input)
        except Exception:
            pass

        # === 1. LEAF VERIFICATION GUARDRAIL ===
        is_leaf, guardrail_reason = validate_leaf_image(image_input)
        if not is_leaf:
            return {
                "is_leaf": False,
                "class_key": "not_a_leaf",
                "title": "Not a Plant Leaf Photo",
                "badge_color": "red",
                "confidence": 0.0,
                "description": "Non-plant image identified.",
                "symptoms": guardrail_reason,
                "remedy": "Please capture a clear, focused photo of a plant leaf in good daylight.",
                "probabilities": {k: 0.0 for k in CLASS_NAMES},
                "image_rgb": image_input.convert("RGB")
            }

        # === 2. MODEL INFERENCE ===
        image_rgb = image_input.convert("RGB")
        image_resized = image_rgb.resize((224, 224))
        img_array = np.expand_dims(np.array(image_resized, dtype=np.float32), axis=0)

        if self.mode == "tflite":
            self.interpreter.set_tensor(self.input_details[0]["index"], img_array)
            self.interpreter.invoke()
            raw_predictions = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        else:
            raw_predictions = self.model.predict(img_array, verbose=0)[0]

        top_index = int(np.argmax(raw_predictions))
        predicted_class = CLASS_NAMES[top_index]
        confidence_percent = float(raw_predictions[top_index] * 100)

        # Out-of-distribution guardrail: if top confidence < 45% or predicted 'unknown'
        if predicted_class == "unknown" or confidence_percent < 45.0:
            return {
                "is_leaf": False,
                "class_key": "inconclusive",
                "title": "Unrecognized / Inconclusive Leaf",
                "badge_color": "orange",
                "confidence": confidence_percent,
                "description": "The leaf features are too blurred or do not match known healthy or deficiency symptoms.",
                "symptoms": "Low confidence pattern match. Image may be out-of-focus, blurry, or showing an unsupported crop species.",
                "remedy": "Please take a sharper, closer photo of a single leaf under natural daylight.",
                "probabilities": {
                    CLASS_NAMES[i]: float(raw_predictions[i] * 100)
                    for i in range(len(CLASS_NAMES))
                },
                "image_rgb": image_rgb
            }

        probabilities = {
            CLASS_NAMES[i]: float(raw_predictions[i] * 100)
            for i in range(len(CLASS_NAMES))
        }

        info = CLASS_DETAILS.get(predicted_class, CLASS_DETAILS["unknown"])

        return {
            "is_leaf": True,
            "class_key": predicted_class,
            "title": info["title"],
            "badge_color": info["badge_color"],
            "confidence": confidence_percent,
            "description": info["description"],
            "symptoms": info["symptoms"],
            "remedy": info["remedy"],
            "probabilities": probabilities,
            "image_rgb": image_rgb
        }


def get_predictor():
    return PlantPredictor()
