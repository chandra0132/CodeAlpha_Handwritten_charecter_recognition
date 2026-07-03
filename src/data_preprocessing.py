import numpy as np
import tensorflow as tf
# pyrefly: ignore [missing-import]
from tensorflow.keras.utils import to_categorical
import ssl

# Bypass SSL certificate verification for downloads (common macOS python issue)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# EMNIST Balanced dataset class mapping
# 0-9: '0'-'9'
# 10-35: 'A'-'Z'
# 36-46: lowercase letters that look distinct from uppercase ('a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't')
EMNIST_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

# MNIST class mapping
MNIST_CLASSES = [str(i) for i in range(10)]

def load_and_preprocess_dataset(dataset_name='mnist'):
    """
    Loads, normalizes, reshapes, and one-hot encodes the dataset.
    Returns:
        x_train, y_train, x_test, y_test, class_names
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        print("Loading MNIST dataset...")
        # pyrefly: ignore [missing-import]
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        class_names = MNIST_CLASSES
        num_classes = 10
        
    elif dataset_name == 'emnist':
        print("Loading EMNIST Balanced dataset...")
        try:
            from emnist import extract_training_samples, extract_test_samples
            x_train, y_train = extract_training_samples('balanced')
            x_test, y_test = extract_test_samples('balanced')
            class_names = EMNIST_CLASSES
            num_classes = 47
        except Exception as e:
            print(f"Warning: Failed to load EMNIST via 'emnist' package ({e}).")
            try:
                print("Attempting to fetch EMNIST Balanced from OpenML mirror...")
                from sklearn.datasets import fetch_openml
                from sklearn.model_selection import train_test_split
                
                emnist_data = fetch_openml(name='EMNIST_Balanced', version=1, parser='auto', as_frame=False)
                X, y = emnist_data.data, emnist_data.target
                
                X = X.reshape(-1, 28, 28)
                X = np.transpose(X, (0, 2, 1))
                y = y.astype(int)
                
                x_train, x_test, y_train, y_test = train_test_split(
                    X, y, test_size=18800, stratify=y, random_state=42
                )
                
                class_names = EMNIST_CLASSES
                num_classes = 47
                print("Successfully loaded EMNIST Balanced from OpenML!")
            except Exception as e_openml:
                print(f"Warning: Failed to fetch from OpenML ({e_openml}). Falling back to MNIST.")
                # pyrefly: ignore [missing-import]
                from tensorflow.keras.datasets import mnist
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
                class_names = MNIST_CLASSES
                num_classes = 10
                dataset_name = 'mnist'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'mnist' or 'emnist'.")
        
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape images to (28, 28, 1) for CNN input
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    # One-hot encode the labels
    y_train_encoded = to_categorical(y_train, num_classes=num_classes)
    y_test_encoded = to_categorical(y_test, num_classes=num_classes)
    
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Train images shape: {x_train.shape}, labels shape: {y_train_encoded.shape}")
    print(f"Test images shape: {x_test.shape}, labels shape: {y_test_encoded.shape}")
    print(f"Number of classes: {num_classes}")
    
    return x_train, y_train_encoded, x_test, y_test_encoded, class_names
