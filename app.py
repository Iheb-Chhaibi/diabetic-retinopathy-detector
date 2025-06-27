from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from tensorflow.keras.losses import BinaryCrossentropy

app = Flask(__name__)

# Custom deserialization for BinaryCrossentropy to handle the 'fn' issue
def custom_loss_fn(**kwargs):
    kwargs.pop('fn', None)  # Remove unsupported argument if it exists
    return BinaryCrossentropy(**kwargs)

# Load the pre-trained model
try:
    model = load_model('DR_noDR.h5', custom_objects={'BinaryCrossentropy': custom_loss_fn})
except Exception as e:
    print(f"Error loading the model: {e}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['file']
        
        # Use BytesIO to handle the file in-memory
        img_bytes = BytesIO(file.read())
        
        # Load the image from BytesIO and preprocess
        img = image.load_img(img_bytes, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image

        # Make a prediction
        prediction = model.predict(img_array)

        # Determine the predicted class (0 or 1)
        predicted_label = "Diabetic Retina" if prediction[0][0] > 0.5 else "No Issues Detected"

        return render_template('index.html', prediction=predicted_label)

    except Exception as e:
        return f"Error processing the image: {e}"

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5555)
