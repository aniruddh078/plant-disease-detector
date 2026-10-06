import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Suppress unnecessary TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

CLASS_NAMES = ["healthy", "nitrogen", "potassium", "unknown"]

CLASS_DETAILS = {
    "healthy": {
        "title": "Healthy Plant Leaf",
        "badge_color": "green",
        "description": "The plant leaf displays uniform green pigmentation, normal cell structure, and balanced nutrient levels.",
        "symptoms": "Uniform leaf coloration, no chlorosis or necrosis, healthy foliage vigor.",
        "remedy": "Continue routine balanced irrigation, adequate sunlight, and standard organic fertilization. No corrective treatment required."
    },
    "nitrogen": {
        "title": "Nitrogen (N) Deficiency",
        "badge_color": "orange",
        "description": "Nitrogen is essential for chlorophyll synthesis and vegetative foliage development.",
        "symptoms": "General yellowing (chlorosis) initiating on older lower leaves, stunted plant growth, pale green new foliage.",
        "remedy": "Apply nitrogen-rich fertilizer such as Urea, Ammonium Nitrate, Calcium Ammonium Nitrate (CAN), or well-composted organic manure."
    },
    "potassium": {
        "title": "Potassium (K) Deficiency",
        "badge_color": "red",
        "description": "Potassium governs stomatal regulation, osmotic balance, and disease resistance.",
        "symptoms": "Marginal chlorosis and browning/scorching along leaf tips and outer edges ('leaf burn'), curling leaves, weakened stems.",
        "remedy": "Apply potassium-rich fertilizer such as Muriate of Potash (MOP / Potassium Chloride), Potassium Sulfate (SOP), or organic wood ash."
    },
    "unknown": {
        "title": "Inconclusive / Unknown Pattern",
        "badge_color": "gray",
        "description": "The image features do not reliably match known healthy or nutrient deficiency profiles.",
        "symptoms": "Blurred photo, non-leaf object, or complex compound deficiency pattern.",
        "remedy": "Please capture a clear, well-focused close-up photo of a single leaf under good lighting and re-scan."
    }
}


class PlantPredictor:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        tflite_path = os.path.join(base_dir, "plant_model.tflite")
        h5_path = os.path.join(base_dir, "plant_model_deploy.h5")
        keras_path = os.path.join(base_dir, "plant_model.keras")

        # Prefer lightweight TFLite (10.6MB) -> H5 -> Keras
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
            raise FileNotFoundError("No model file (plant_model.tflite, plant_model_deploy.h5, or plant_model.keras) found.")

    def predict(self, image_input):
        if not isinstance(image_input, Image.Image):
            image_input = Image.open(image_input)

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
