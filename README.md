# NeuroWrite AI: Handwritten Character Recognition Dashboard

A professional, internship-level Deep Learning project built using Python and TensorFlow/Keras. This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits (MNIST) and alphanumeric characters (EMNIST Balanced), featuring a CLI pipeline, a Jupyter Notebook walkthrough, and an interactive Flask-based web dashboard with a real-time drawing canvas and multi-character OCR text segmenter.

---

## 🚀 Key Features

1. **Flexible Datasets**: Built-in loaders for both **MNIST** (10 digit classes) and **EMNIST Balanced** (47 alphanumeric classes, comprising 0-9, A-Z, and distinct lowercase characters).
2. **Robust CNN Architecture**: Optimized layer stack containing 2D Convolutions, Max Pooling, Dropout (for regularization), L2 weight penalization, and a Dense Softmax classifier.
3. **Advanced Preprocessing Pipeline**: Standardizes custom drawings or uploads using aspect-ratio preserving scaling, binarization, bounding box cropping, and centering on a 28x28 pixel grid matching the dataset distribution.
4. **Interactive Drawing Canvas**: Draw digits or characters with touch/mouse support on the screen and get real-time predictions and top-5 probability scores.
5. **Horizontal OCR Character Segmenter**: Draw multiple letters or a whole word, and the system automatically crops and isolates each letter contour, performs horizontal sorting, and transcribes the combined text.
6. **Drag & Drop Upload Center**: Upload a scan or photo of handwritten characters and see immediate thresholding and classification.
7. **EDA & Model Metrics Panel**: Detailed dashboard plots of learning curves (Accuracy/Loss progress) and Seaborn-based confusion matrix heatmaps.

---

## 📁 Project Structure

```
charecter_recognition/
├── app.py                   # Flask server backend (web dashboard)
├── requirements.txt         # Project package requirements
├── README.md                # Extensive setup and usage guide
├── models/                  # Saved models (.keras) and metrics JSONs
│   ├── mnist_model.keras
│   ├── emnist_model.keras
│   └── *_history.json
├── src/                     # Source modules
│   ├── __init__.py
│   ├── data_preprocessing.py # Preprocessor, normalizer, and SSL bypass
│   ├── model.py             # Keras CNN Architecture
│   ├── train.py             # Model training CLI
│   ├── evaluate.py          # Model evaluation CLI & heatmap generator
│   └── predict.py           # Command-line prediction tool
├── templates/               # Flask UI templates
│   └── index.html           # Main dashboard markup
├── static/                  # Web dashboard styling and assets
│   ├── css/
│   │   └── style.css        # Premium dark glassmorphism stylesheet
│   ├── js/
│   │   └── main.js          # Canvas controller, AJAX handlers, top-5 charts
│   └── assets/              # Generated learning curves, confusion matrix, placeholder
│       ├── placeholder_graph.png
│       ├── mnist_accuracy.png
│       └── ...
└── notebooks/               # Jupyter exploration folder
    └── exploration_and_training.ipynb  # Step-by-step EDA & training walkthrough
```

---

## 🛠️ Setup and Installation

### 1. Prerequisites
Make sure Python 3.10+ and `pip` are installed on your machine.

### 2. Install Dependencies
Install all required libraries via pip:
```bash
pip install -r requirements.txt
```

---

## 🏋️ Model Training & Evaluation

The training scripts support two datasets: `mnist` (digits 0-9) or `emnist` (alphanumeric letters & digits).

### 1. Train the CNN Model
To train the model on the EMNIST Balanced dataset (trained for 10 epochs by default):
```bash
python3 src/train.py --dataset emnist --epochs 10 --batch_size 128
```
For the simpler MNIST dataset:
```bash
python3 src/train.py --dataset mnist --epochs 5
```
*Note: Training will save the model file (e.g. `models/mnist_model.keras`) and automatically generate learning curve graphs under `static/assets/`.*

### 2. Evaluate Model Performance
To generate the detailed classification metrics (Precision, Recall, F1-score, and Confusion Matrix Heatmap):
```bash
python3 src/evaluate.py --model models/emnist_model.keras --dataset emnist
```
This prints the metrics to stdout, saves a text summary in `models/`, and writes the Confusion Matrix heatmap image inside `static/assets/`.

### 3. Run Command-Line Inference
Run classification on any custom image of a handwritten character:
```bash
python3 src/predict.py --model models/mnist_model.keras --image path/to/sample.png --dataset mnist
```

---

## 🌐 Running the Web Dashboard

Launch the Flask development server:
```bash
python3 app.py
```
Open your browser and navigate to **`http://127.0.0.1:5000`**.

### Dashboard Interface Guide
*   **Drawing Board Tab**: Select the active pipeline (MNIST or EMNIST) in the header. Draw a single character, and check the neon prediction circle and the top-5 confidence bars.
*   **Upload Predictor Tab**: Drop an image file inside the dotted frame. The server handles grayscale thresholding and displays predictions immediately.
*   **Word OCR Tab**: Draw letters or a word side-by-side. Click **Perform OCR**. The backend segments the canvas using contours, sorts segments from left to right, runs predictions on each letter, and transcribes the text.
*   **EDA & Performance Tab**: Inspect the neural network layers, dataset composition tables, and view training curves / confusion matrices for the active pipeline.
