from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = "vgg19_model_improved.h5"
CLASS_NAMES = ["Diabetic Retinopathy", "Glaucoma", "Normal", "Cataract"]
INPUT_SHAPE = (224, 224, 3)

# Global model variable
model = None

def load_model():
    """Load the VGG19 model with error handling."""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        logger.info("Loading VGG19 model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction."""
    try:
        # Convert to RGB if image is in a different color space
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to match model input size
        image = image.resize((INPUT_SHAPE[0], INPUT_SHAPE[1]))
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32)
        
        # Normalize pixel values
        image_array = image_array / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load model before creating routes
if not load_model():
    logger.error("Failed to load model. Server may not function correctly.")

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    try:
        status = {
            "status": "healthy",
            "model_loaded": model is not None,
            "timestamp": datetime.now().isoformat(),
            "model_path": MODEL_PATH,
            "input_shape": INPUT_SHAPE,
            "classes": CLASS_NAMES
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image prediction requests."""
    try:
        # Check if model is loaded
        if model is None:
            logger.error("Model not loaded")
            return jsonify({"error": "Model not loaded"}), 500

        # Validate request
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if not file.filename.lower().endswith(tuple(allowed_extensions)):
            return jsonify({"error": "Invalid file type. Please upload a PNG, JPG, or JPEG image."}), 400

        # Read and process the image
        try:
            image = Image.open(io.BytesIO(file.read()))
        except Exception as e:
            logger.error(f"Error reading image: {str(e)}")
            return jsonify({"error": "Invalid image file"}), 400

        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Prepare response
        response = {
            "class": predicted_class,
            "accuracy": f"{confidence * 100:.2f}%",
            "confidence_scores": {
                class_name: f"{float(score) * 100:.2f}%"
                for class_name, score in zip(CLASS_NAMES, predictions[0])
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Prediction successful: {predicted_class} ({confidence * 100:.2f}%)")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)