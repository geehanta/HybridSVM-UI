import os
from flask import Flask, render_template, request, redirect, flash, jsonify
from werkzeug.utils import secure_filename
import joblib
import cv2
import csv
from datetime import datetime
import numpy as np

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

# Load model
model = joblib.load("model/hybrid_model.joblib")

# Utility: check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Utility: preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = image.flatten()
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = preprocess_image(filepath)
            prediction = model.predict(img)[0]

            result = 'Cancerous' if prediction == 1 else 'Benign'
            return render_template('index.html', prediction=result, uploaded=True, image=filename)

    return render_template('index.html', uploaded=False)

@app.route('/submit-rating', methods=['POST'])
def submit_rating():
    rating = request.form.get('rating')
    if rating in ['0', '1']:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open('model_feedback.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, rating])
        return jsonify({'message': 'Thanks for your feedback!'})
    return jsonify({'message': 'Invalid rating.'})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)