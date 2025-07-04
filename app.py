import os
import csv
import joblib
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, flash, jsonify, session
from werkzeug.utils import secure_filename

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'model'
MODEL_FILENAME = 'hybrid_model.joblib'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'

# --- Ensure folders exist ---
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

model = None

# --- Utilities ---
def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def load_model():
    global model
    model_path = os.path.join(MODEL_FOLDER, MODEL_FILENAME)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = None

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = image.flatten()
    return np.expand_dims(image, axis=0)

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    global model
    model_path = os.path.join(MODEL_FOLDER, MODEL_FILENAME)

    # Handle model upload
    if 'model_file' in request.files:
        file = request.files['model_file']
        if file.filename.endswith('.joblib'):
            file.save(model_path)
            load_model()
            session['model_just_uploaded'] = True  # ✅ flash trigger
            return redirect('/')
        else:
            flash("Please upload a valid .joblib file.")
            return redirect('/')

    # Load model if already uploaded
    model_loaded = os.path.exists(model_path)
    if model_loaded and model is None:
        load_model()

    # One-time toast after model upload
    model_just_uploaded = session.pop('model_just_uploaded', False)

    prediction_result = None
    uploaded = False
    image_name = None

    # Handle image prediction
    if request.method == 'POST' and 'image' in request.files and model_loaded:
        file = request.files['image']
        if file.filename == '':
            flash('No image selected.')
            return redirect('/')

        if allowed_image(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            img = preprocess_image(filepath)
            prediction = model.predict(img)[0]
            prediction_result = 'Cancerous' if prediction == 1 else 'Benign'
            uploaded = True
            image_name = filename

    return render_template(
        'index.html',
        model_loaded=model_loaded,
        model_just_uploaded=model_just_uploaded,
        prediction=prediction_result,
        uploaded=uploaded,
        image=image_name
    )

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

# --- Run ---
if __name__ == '__main__':
    import os

    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
