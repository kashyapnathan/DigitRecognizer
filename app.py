from PIL import Image, ImageOps
import io, logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import render_template, request, flash

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Loading model...")
digit_model = load_model('digit_model.h5')
logger.info("Model loaded successfully!")

@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    try:
        logger.info("Received request for /predict_digit")
        image_data = request.files['image'].read()
        print("Image read successfully")
        
        # Convert the image data to an Image object
        image = Image.open(io.BytesIO(image_data))
        
        # Convert the image to grayscale
        image = ImageOps.grayscale(image)
        
        # Resize the image to 28x28 pixels
        image = image.resize((28, 28))
        
        # Now, convert the Image object to an array
        image = img_to_array(image)
        print("Converted image to array")
        
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        print("Preprocessed the image")
        
        prediction = digit_model.predict(image)
        print(f"Raw prediction probabilities: {prediction[0]}")
        predicted_digit = np.argmax(prediction)
        confidence_value = prediction[0][predicted_digit]
        print(f"Predicted digit: {predicted_digit} with confidence: {confidence_value:.2f}")
        
        return jsonify({'predicted_digit': int(predicted_digit), 'confidence': float(confidence_value)})
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image'].read()
        image = img_to_array(image)
        
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        
        prediction = digit_model.predict(image)
        predicted_digit = np.argmax(prediction)
        
        flash(f"Predicted Digit: {predicted_digit}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
