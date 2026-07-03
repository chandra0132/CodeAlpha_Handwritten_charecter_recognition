import os
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template

# Deferred library imports for TFLite/Keras to enable instant startup
TFLITE_AVAILABLE = None
KERAS_AVAILABLE = None
tflite = None
tf = None

def init_model_libs():
    global TFLITE_AVAILABLE, KERAS_AVAILABLE, tflite, tf
    if TFLITE_AVAILABLE is not None:
        return
        
    try:
        # pyrefly: ignore [missing-import]
        import tflite_runtime.interpreter as tfl_lib
        tflite = tfl_lib
        TFLITE_AVAILABLE = True
    except ImportError:
        try:
            import tensorflow.lite as tfl_lib
            tflite = tfl_lib
            TFLITE_AVAILABLE = True
        except ImportError:
            TFLITE_AVAILABLE = False

    try:
        import tensorflow as tf_lib
        tf = tf_lib
        KERAS_AVAILABLE = True
    except ImportError:
        KERAS_AVAILABLE = False

# Ensure modules in src are importable
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import preprocess_image
from src.data_preprocessing import EMNIST_CLASSES, MNIST_CLASSES

app = Flask(__name__)

# Cache models in memory to avoid reloading from disk on each request
MODELS_CACHE = {}
MODEL_DIR = "models"

def get_model(dataset_name):
    """
    Lazy-load and cache model from models directory (TFLite first, Keras as fallback).
    """
    init_model_libs()
    dataset_name = dataset_name.lower()
    if dataset_name not in MODELS_CACHE:
        tflite_path = os.path.join(MODEL_DIR, f"{dataset_name}_model.tflite")
        keras_path = os.path.join(MODEL_DIR, f"{dataset_name}_model.keras")
        
        if TFLITE_AVAILABLE and os.path.exists(tflite_path):
            print(f"Loading TFLite {dataset_name.upper()} model into memory cache...")
            interpreter = tflite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            MODELS_CACHE[dataset_name] = {
                'type': 'tflite',
                'interpreter': interpreter,
                'input_details': interpreter.get_input_details(),
                'output_details': interpreter.get_output_details()
            }
        elif KERAS_AVAILABLE and os.path.exists(keras_path):
            print(f"Loading Keras {dataset_name.upper()} model into memory cache...")
            MODELS_CACHE[dataset_name] = {
                'type': 'keras',
                'model': tf.keras.models.load_model(keras_path)
            }
        else:
            raise FileNotFoundError(
                f"No suitable model found for {dataset_name.upper()} (TFLite or Keras files not found/unsupported)."
            )
    return MODELS_CACHE[dataset_name]

def parse_base64_image(base64_str):
    """
    Decodes base64 string from canvas/AJAX into OpenCV image array.
    """
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route('/')
def home():
    """
    Main page showing the interactive dashboard.
    """
    # Check if models are available (either Keras or TFLite), so we can notify the user in the UI
    models_available = {
        'mnist': os.path.exists(os.path.join(MODEL_DIR, 'mnist_model.keras')) or os.path.exists(os.path.join(MODEL_DIR, 'mnist_model.tflite')),
        'emnist': os.path.exists(os.path.join(MODEL_DIR, 'emnist_model.keras')) or os.path.exists(os.path.join(MODEL_DIR, 'emnist_model.tflite'))
    }
    return render_template('index.html', models_available=models_available)

@app.route('/predict', methods=['POST'])
def predict_single():
    """
    Predicts a single handwritten character from canvas data or file upload.
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        dataset = data.get('dataset', 'mnist').lower()
        image_data = data['image']
        
        # Load correct classes mapping
        classes = EMNIST_CLASSES if dataset == 'emnist' else MNIST_CLASSES
        
        # Load cached model
        try:
            model = get_model(dataset)
        except FileNotFoundError as e:
            return jsonify({'error': str(e), 'not_trained': True}), 400
            
        # Parse base64 image
        img = parse_base64_image(image_data)
        
        # Preprocess using same rules as CLI predict.py
        preprocessed_img, gray_img = preprocess_image(img, invert_check=True)
        
        # Make batch input
        input_data = np.expand_dims(preprocessed_img, axis=0)
        
        # Predict
        if model['type'] == 'tflite':
            interpreter = model['interpreter']
            input_details = model['input_details']
            output_details = model['output_details']
            
            input_data = input_data.astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]
        else:
            preds = model['model'].predict(input_data)[0]
        pred_idx = np.argmax(preds)
        confidence = float(preds[pred_idx])
        pred_char = classes[pred_idx]
        
        # Get top-5 predictions
        top_5_indices = np.argsort(preds)[::-1][:5]
        top_5 = []
        for idx in top_5_indices:
            top_5.append({
                'class': classes[idx],
                'confidence': float(preds[idx])
            })
            
        return jsonify({
            'success': True,
            'prediction': pred_char,
            'confidence': confidence,
            'top_5': top_5
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500

@app.route('/predict_ocr', methods=['POST'])
def predict_ocr():
    """
    Segments a drawn word or multi-character sentence horizontally,
    predicts each individual character, and returns the combined string.
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        dataset = data.get('dataset', 'emnist').lower() # EMNIST is better for OCR words
        image_data = data['image']
        
        classes = EMNIST_CLASSES if dataset == 'emnist' else MNIST_CLASSES
        
        try:
            model = get_model(dataset)
        except FileNotFoundError as e:
            return jsonify({'error': str(e), 'not_trained': True}), 400
            
        img = parse_base64_image(image_data)
        
        # Preprocess to grayscale and threshold to detect characters
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Border check to invert if background is white
        h_img, w_img = thresh.shape
        border_pixels = np.concatenate([thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]])
        if np.mean(border_pixels) > 127:
            thresh = cv2.bitwise_not(thresh)
            
        # Find contours (bounding boxes of characters)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return jsonify({
                'success': True,
                'text': '',
                'characters': []
            })
            
        # Filter noise & sort contours horizontally (left to right)
        valid_contours = []
        for ctr in contours:
            x, y, w_box, h_box = cv2.boundingRect(ctr)
            # Area/dimension filtering to discard single random dots
            if w_box > 4 and h_box > 4 and cv2.contourArea(ctr) > 15:
                valid_contours.append((x, y, w_box, h_box, ctr))
                
        # Sort left-to-right based on x-coordinate
        valid_contours = sorted(valid_contours, key=lambda c: c[0])
        
        # Segment, preprocess, and predict each character
        predictions = []
        combined_text = []
        
        # We also need to check for spaces between characters.
        # If the gap between consecutive bounding boxes is larger than a threshold, add a space.
        prev_x_end = None
        
        for idx, (x, y, w_box, h_box, ctr) in enumerate(valid_contours):
            # Check for space (if gap is greater than 1.2 * average character width or 40 pixels)
            if prev_x_end is not None:
                gap = x - prev_x_end
                # Average character width so far
                widths = [item[2] for item in valid_contours]
                avg_width = np.mean(widths)
                if gap > max(35, avg_width * 0.8):
                    combined_text.append(' ')
                    predictions.append({
                        'char': ' ',
                        'confidence': 1.0,
                        'is_space': True,
                        'bbox': [int(prev_x_end), int(y), int(gap), int(h_box)]
                    })
            
            # Crop single character
            char_crop = thresh[y:y+h_box, x:x+w_box]
            
            # Pad and center to 28x28
            if w_box > h_box:
                sf = 20.0 / w_box
                new_w = 20
                new_h = int(h_box * sf)
            else:
                sf = 20.0 / h_box
                new_h = 20
                new_w = int(w_box * sf)
                
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            
            resized = cv2.resize(char_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            canvas = np.zeros((28, 28), dtype=np.uint8)
            x_offset = (28 - new_w) // 2
            y_offset = (28 - new_h) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            # Normalize and batch
            preprocessed_char = canvas.astype('float32') / 255.0
            preprocessed_char = np.expand_dims(preprocessed_char, axis=-1)
            input_char = np.expand_dims(preprocessed_char, axis=0)
            
            # Run prediction
            if model['type'] == 'tflite':
                interpreter = model['interpreter']
                input_details = model['input_details']
                output_details = model['output_details']
                
                input_char = input_char.astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_char)
                interpreter.invoke()
                preds = interpreter.get_tensor(output_details[0]['index'])[0]
            else:
                preds = model['model'].predict(input_char)[0]
            pred_idx = np.argmax(preds)
            confidence = float(preds[pred_idx])
            pred_char = classes[pred_idx]
            
            combined_text.append(pred_char)
            predictions.append({
                'char': pred_char,
                'confidence': confidence,
                'is_space': False,
                'bbox': [int(x), int(y), int(w_box), int(h_box)]
            })
            
            prev_x_end = x + w_box
            
        full_string = "".join(combined_text)
        
        return jsonify({
            'success': True,
            'text': full_string,
            'characters': predictions
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Default Flask port is 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
