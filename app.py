from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import logging
import os
from huggingface_hub import hf_hub_download 

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO) #
logger = logging.getLogger(__name__)

MODEL_REPO_ID = "Samyuktha-Rama/fingerprint-bloodgroup-classifier"
MODEL_FILENAME = "blood_group_model.h5"

try:
    logger.info(f"Attempting to download model '{MODEL_FILENAME}' from Hugging Face Hub (Repo: {MODEL_REPO_ID})...")
    
    downloaded_model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    logger.info(f"Model successfully downloaded to: {downloaded_model_path}")

    model = load_model(downloaded_model_path)
    logger.info("Model loaded successfully into Keras")
except Exception as e:
    logger.error(f"FATAL ERROR: Could not load model from Hugging Face Hub. Please check MODEL_REPO_ID and file name. Error: {e}")

    exit(1)

blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

def is_fingerprint(image):
    """
    Checks if an uploaded image likely contains a fingerprint pattern 
    based on edge density, variance, and frequency analysis.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)

        edge_count = np.sum(edges > 0)
        total_pixels = edges.size
        edge_percentage = (edge_count / total_pixels) * 100

        min_edge_percent = 2.5 
        max_edge_percent = 30.0 
        threshold = total_pixels * (min_edge_percent / 100)

        variance = np.var(gray)
        
        h, w = gray.shape
        roi = gray[h//4:3*h//4, w//4:3*w//4]
        freq = np.fft.fft2(roi)
        freq_power = np.abs(freq) ** 2
        ridge_freq = np.mean(freq_power) > 500

        logger.debug(f"Edge %: {edge_percentage:.2f}% (Range: {min_edge_percent}-{max_edge_percent})")
        logger.debug(f"Variance: {variance:.2f} (Min: 50)")
        logger.debug(f"Ridge power mean: {np.mean(freq_power):.2f} (Min: 500)")

        is_fp = (edge_count > threshold and 
                 min_edge_percent <= edge_percentage <= max_edge_percent and 
                 variance > 50 and 
                 ridge_freq)

        if not is_fp:
            logger.warning(f"Image rejected. Edge%: {edge_percentage:.2f}, Var: {variance:.2f}, Ridge: {ridge_freq}")
            
        return is_fp
    except Exception as e:
        logger.error(f"Error in is_fingerprint: {e}")
        return False

@app.route('/predict', methods=['POST'])
def predict_blood_group():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    try:
        file_data = file.read()
        
        img_rgb = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
        if img_rgb is None:
            return jsonify({'error': 'Invalid image format or corrupted file'}), 400
        
        if not is_fingerprint(img_rgb):
            return jsonify({'error': 'Uploaded image is not a recognizable fingerprint'}), 400
        logger.info("Fingerprint verification passed.")

        img_gray = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        
        img = cv2.resize(img_gray, (128, 128))
        img = img.reshape(1, 128, 128, 1) / 255.0
        
        prediction = model.predict(img, verbose=0)
        confidence = np.max(prediction)
        blood_group = blood_groups[np.argmax(prediction)]
        
        logger.info(f"Predicted blood group: {blood_group}, Confidence: {confidence:.4f}")
        
        return jsonify({'blood_group': blood_group})
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000)) 
    app.run(debug=True, host='0.0.0.0', port=port)