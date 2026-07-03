import os
import argparse
import numpy as np
import cv2

# Ensure modules in src are importable
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import EMNIST_CLASSES, MNIST_CLASSES

def preprocess_image(image_path_or_array, invert_check=True):
    """
    Preprocesses a custom handwriting image to match the format of MNIST/EMNIST:
    - Grayscale conversion
    - Thresholding / Binarization
    - Auto-inverting so character is white and background is black
    - Bounding-box cropping
    - Resizing to fit 20x20 aspect-ratio-preserved box
    - Centering on a 28x28 canvas (4px padding)
    - Normalizing pixel values to [0, 1]
    
    Args:
        image_path_or_array (str or np.ndarray): File path or pre-loaded image array.
        invert_check (bool): Whether to auto-check if background needs inverting.
        
    Returns:
        preprocessed_img (np.ndarray): Preprocessed image array of shape (28, 28, 1).
        raw_gray (np.ndarray): Intermediary grayscale image for debugging.
    """
    # 1. Load image
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise FileNotFoundError(f"Could not load image from path: {image_path_or_array}")
    else:
        img = image_path_or_array.copy()
        
    # 2. Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    # 3. Clean and threshold (Otsu binarization)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # We binarize using OTSU's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Auto-inversion check:
    # If the user drew dark-on-light, THRESH_BINARY_INV turns it white-on-black (correct).
    # If the user drew light-on-dark (e.g. white on black canvas), THRESH_BINARY_INV makes it black-on-white.
    # We check the border pixel mean to determine background.
    if invert_check:
        h, w = thresh.shape
        border_pixels = np.concatenate([
            thresh[0, :],          # Top row
            thresh[-1, :],         # Bottom row
            thresh[:, 0],          # Left col
            thresh[:, -1]          # Right col
        ])
        # If the borders are mostly white (>50% mean), then the background is white.
        # We invert the thresholded image so the background becomes black and character becomes white.
        if np.mean(border_pixels) > 127:
            thresh = cv2.bitwise_not(thresh)
            
    # 4. Find bounding box of the digit/character
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour by area
        c = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(c)
        
        # Crop the character region
        cropped = thresh[y:y+h_box, x:x+w_box]
        
        # Determine scaling factors (resize to fit inside 20x20 box preserving aspect ratio)
        if w_box > h_box:
            sf = 20.0 / w_box
            new_w = 20
            new_h = int(h_box * sf)
        else:
            sf = 20.0 / h_box
            new_h = 20
            new_w = int(w_box * sf)
            
        # Avoid zero dimensions
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create empty 28x28 black canvas
        canvas = np.zeros((28, 28), dtype=np.uint8)
        
        # Center the resized character on the canvas
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        preprocessed = canvas
    else:
        # Fallback: Just resize the thresh to 28x28 if no contours are found
        preprocessed = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
        
    # 5. Normalize pixel values to [0, 1]
    preprocessed = preprocessed.astype('float32') / 255.0
    
    # 6. Expand dimensions to (28, 28, 1)
    preprocessed = np.expand_dims(preprocessed, axis=-1)
    
    return preprocessed, gray

def predict(model_path, image_path, dataset_name='mnist'):
    """
    Loads model, preprocesses image, runs prediction, and displays output.
    """
    import tensorflow as tf
    # Check dataset and get classes
    if dataset_name.lower() == 'emnist':
        classes = EMNIST_CLASSES
    else:
        classes = MNIST_CLASSES
        
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess image
    preprocessed_img, _ = preprocess_image(image_path)
    
    # Add batch dimension -> (1, 28, 28, 1)
    input_data = np.expand_dims(preprocessed_img, axis=0)
    
    # Run prediction
    preds = model.predict(input_data)[0]
    pred_idx = np.argmax(preds)
    confidence = preds[pred_idx]
    pred_char = classes[pred_idx]
    
    print("\n" + "="*40)
    print(f"PREDICTION RESULT ({dataset_name.upper()})")
    print("="*40)
    print(f"Image:            {os.path.basename(image_path)}")
    print(f"Predicted Class:  {pred_char}")
    print(f"Confidence Score: {confidence * 100:.2f}%")
    print("="*40)
    
    # Top 5 predictions
    top_5_idx = np.argsort(preds)[::-1][:5]
    print("\nTop 5 Predictions:")
    for rank, idx in enumerate(top_5_idx, 1):
        print(f"{rank}. Class: '{classes[idx]}' | Confidence: {preds[idx] * 100:.2f}%")
        
    return pred_char, confidence, preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict handwritten digit or character from image.')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained Keras model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image to classify')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'emnist'],
                        help='Dataset model was trained on: mnist or emnist')
    
    args = parser.parse_args()
    
    predict(
        model_path=args.model,
        image_path=args.image,
        dataset_name=args.dataset
    )
