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


class PlantPredictor:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        tflite_path = os.path.join(base_dir, "plant_model.tflite")
        h5_path = os.path.join(base_dir, "plant_model_deploy.h5")
        keras_path = os.path.join(base_dir, "plant_model.keras")

        # Prefer fast, edge-optimized TFLite (10.6MB)
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
            raise FileNotFoundError("No trained model file found (looked for plant_model.tflite, plant_model_deploy.h5, or plant_model.keras).")

    def predict(self, image_input):
        """
        Preprocesses image with phone EXIF orientation correction and runs inference.
        """
        if not isinstance(image_input, Image.Image):
            image_input = Image.open(image_input)

        # Fix mobile phone camera orientation if EXIF tags exist
        try:
            image_input = ImageOps.exif_transpose(image_input)
        except Exception:
            pass

        # Convert safely to RGB
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

        probabilities = {
            CLASS_NAMES[i]: float(raw_predictions[i] * 100)
            for i in range(len(CLASS_NAMES))
        }

        info = CLASS_DETAILS.get(predicted_class, CLASS_DETAILS["unknown"])

        return {
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
