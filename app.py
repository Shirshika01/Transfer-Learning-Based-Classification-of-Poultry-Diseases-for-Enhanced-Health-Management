from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model
model = tf.keras.models.load_model('healthy_vs_rotten.h5')
classes = ['Coccidiosis', 'Healthy', 'Salmonella', 'Newcastle']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction_page')
def prediction_page():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load and preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    return render_template('prediction.html', label=predicted_class, filename=filename)

@app.route('/display/<filename>')
def display(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
