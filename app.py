import os
import csv
import joblib
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, flash, jsonify, session
from werkzeug.utils import secure_filename
from flask import send_file

# --- Configuration ---
UPLOAD_FOLDER  = 'uploads'
MODEL_FOLDER   = 'model'
IMG_SIZE       = (224, 224)   # ResNet50 native resolution
CATEGORIES     = ['colon_aca', 'colon_n']  # index 0 = cancer, index 1 = normal
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Expected model artefact filenames inside MODEL_FOLDER
MODEL_FILES = {
    'svm':       'hybrid_search_model.joblib',
    'scaler':    'scaler.joblib',
    'pca':       'pca.joblib',
    'threshold': 'best_threshold.npy',
}

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER']  = MODEL_FOLDER
app.config['SESSION_TYPE']  = 'filesystem'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER,  exist_ok=True)

# --- Global pipeline objects ---
svm_model         = None
scaler            = None
pca               = None
best_threshold    = 0.5
feature_extractor = None   # ResNet50 – loaded lazily after first model upload


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def all_model_files_present():
    """Return True only when every required artefact exists on disk."""
    return all(
        os.path.exists(os.path.join(MODEL_FOLDER, fname))
        for fname in MODEL_FILES.values()
    )


def load_resnet_extractor():
    """Load ResNet50 feature extractor (GPU/CPU) – called once after artefacts land."""
    global feature_extractor
    if feature_extractor is not None:
        return

    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Model

    print('[INFO] Loading ResNet50 feature extractor...')
    base = ResNet50(weights='imagenet', include_top=False,
                    pooling='avg', input_shape=(*IMG_SIZE, 3))
    feature_extractor = Model(inputs=base.input, outputs=base.output)
    feature_extractor.trainable = False
    print('[INFO] ResNet50 ready.')


def load_pipeline():
    """Load all four artefacts from MODEL_FOLDER into global variables."""
    global svm_model, scaler, pca, best_threshold

    if not all_model_files_present():
        return False

    svm_model      = joblib.load(os.path.join(MODEL_FOLDER, MODEL_FILES['svm']))
    scaler         = joblib.load(os.path.join(MODEL_FOLDER, MODEL_FILES['scaler']))
    pca            = joblib.load(os.path.join(MODEL_FOLDER, MODEL_FILES['pca']))
    threshold_arr  = np.load(os.path.join(MODEL_FOLDER, MODEL_FILES['threshold']))
    best_threshold = float(threshold_arr[0])
    best_threshold = max(best_threshold, 0.2)  # Avoid too-low thresholds that would cause over-prediction of cancer
    
    load_resnet_extractor()
    print(f'[INFO] Pipeline loaded. Threshold = {best_threshold:.4f}')
    return True


def extract_features(image_path):
    """
    Full preprocessing pipeline matching the notebook:
      1. Load image as RGB 224x224
      2. ResNet50 preprocess_input
      3. extractor.predict  ->  2048-dim vector
      4. scaler.transform
      5. pca.transform
    Returns array of shape (1, n_pca_components).
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'Could not read image: {image_path}')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img.astype(np.float32), axis=0)
    img = preprocess_input(img)

    feat   = feature_extractor.predict(img, verbose=0)   # (1, 2048)
    feat_s = scaler.transform(feat)                       # standardise
    feat_p = pca.transform(feat_s)                        # reduce dims
    return feat_p


def run_prediction(image_path):
    """
    Returns (label_str, confidence_pct, threshold_used).
    label_str is 'Cancerous' or 'Benign'.
    """
    feat_p = extract_features(image_path)

    prob      = svm_model.predict_proba(feat_p)[0][1]     # P(cancer)
    label     = CATEGORIES[0] if prob >= best_threshold else CATEGORIES[1]
    label_str = 'Cancerous' if label == CATEGORIES[0] else 'Benign'

    return label_str, round(float(prob) * 100, 2), best_threshold


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    global svm_model

    # ── Handle model artefact uploads ────────────────────────────────────────
    if request.method == 'POST' and 'svm_file' in request.files:
        uploads = {
            'svm':       request.files.get('svm_file'),
            'scaler':    request.files.get('scaler_file'),
            'pca':       request.files.get('pca_file'),
            'threshold': request.files.get('threshold_file'),
        }

        errors = []
        for key, f in uploads.items():
            if f is None or f.filename == '':
                errors.append(f'Missing file: {key}')
            elif key != 'threshold' and not f.filename.endswith('.joblib'):
                errors.append(f'"{key}" must be a .joblib file.')
            elif key == 'threshold' and not f.filename.endswith('.npy'):
                errors.append('"threshold" must be a .npy file.')

        if errors:
            flash(' | '.join(errors))
            return redirect('/')

        for key, f in uploads.items():
            dest = os.path.join(MODEL_FOLDER, MODEL_FILES[key])
            f.save(dest)

        success = load_pipeline()
        if success:
            session['model_just_uploaded'] = True
        else:
            flash('Files saved but pipeline failed to load — check filenames/formats.')
        return redirect('/')

    # ── Load existing pipeline on GET (if artefacts already on disk) ─────────
    model_loaded = all_model_files_present()
    if model_loaded and svm_model is None:
        load_pipeline()

    model_just_uploaded = session.pop('model_just_uploaded', False)

    prediction_result = None
    confidence        = None
    threshold_used    = None
    uploaded          = False
    image_name        = None
    error_msg         = None

    # ── Handle image prediction ───────────────────────────────────────────────
    if request.method == 'POST' and 'image' in request.files and model_loaded:
        file = request.files['image']
        if file.filename == '':
            flash('No image selected.')
            return redirect('/')

        if allowed_image(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:
                prediction_result, confidence, threshold_used = run_prediction(filepath)
                uploaded   = True
                image_name = filename
            except Exception as e:
                error_msg = f'Prediction failed: {str(e)}'
        else:
            flash('Unsupported file type. Please upload PNG, JPG or JPEG.')
            return redirect('/')

    return render_template(
        'index.html',
        model_loaded=model_loaded,
        model_just_uploaded=model_just_uploaded,
        prediction=prediction_result,
        confidence=confidence,
        threshold=threshold_used,
        uploaded=uploaded,
        image=image_name,
        error=error_msg,
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

# --- New route to download feedback CSV ---
@app.route('/download-feedback', methods=['GET'])
def download_feedback():
    feedback_file = 'model_feedback.csv'
    if os.path.exists(feedback_file):
        return send_file(feedback_file, as_attachment=True)
    else:
        flash("No feedback file found.")
        return redirect('/')
# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
