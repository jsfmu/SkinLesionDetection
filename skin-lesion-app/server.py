from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Load the model
try:
    # Try to load the model from Fold 2
    model = tf.keras.models.load_model('../best_model_fold_4.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

def preprocess_image(image_data):
    try:
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get image data
        image_file = request.files['image']
        image_data = image_file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        raw_probability = float(prediction[0][0])
        
        # Ensure probability is between 0 and 1
        probability = max(0.0, min(1.0, raw_probability))
        
        # Determine result and confidence
        result = 'Malignant' if probability > 0.5 else 'Benign'
        confidence = probability if probability > 0.5 else 1 - probability
        
        # Generate recommendations based on confidence
        if confidence < 0.6:
            recommendation = "Follow-up Recommended: Low confidence in prediction. Please consult a healthcare professional."
        elif confidence < 0.8:
            recommendation = "Moderate confidence. Consider professional evaluation for confirmation."
        else:
            recommendation = "High confidence in prediction. Still recommended to consult a healthcare professional for final diagnosis."
        
        return jsonify({
            'prediction': result,
            'confidence': f"{confidence * 100:.2f}",
            'recommendation': recommendation
        })
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 