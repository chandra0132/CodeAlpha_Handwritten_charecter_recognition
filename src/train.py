import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Ensure modules in src are importable
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_and_preprocess_dataset
from src.model import build_cnn_model

def train(dataset_name='mnist', epochs=10, batch_size=128, model_dir='models', assets_dir='static/assets'):
    """
    Trains the CNN model on the specified dataset and saves results.
    """
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    # 1. Load and preprocess dataset
    x_train, y_train, x_test, y_test, class_names = load_and_preprocess_dataset(dataset_name)
    num_classes = len(class_names)
    
    # 2. Build model
    model = build_cnn_model(input_shape=(28, 28, 1), num_classes=num_classes)
    model.summary()
    
    # Define filenames
    model_path = os.path.join(model_dir, f"{dataset_name}_model.keras")
    history_path = os.path.join(model_dir, f"{dataset_name}_history.json")
    
    # 3. Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # 4. Train model
    print(f"Starting training on {dataset_name.upper()} for {epochs} epochs...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    # Save best model explicitly if checkpoint didn't save (e.g. if training finished without trigger)
    if not os.path.exists(model_path):
        model.save(model_path)
        print(f"Saved model to {model_path}")
        
    # 5. Save history to JSON
    # Convert history values to float for JSON serialization
    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
    print(f"Saved training history to {history_path}")
    
    # 6. Generate and save training graphs
    plot_and_save_curves(history.history, dataset_name, assets_dir)
    
    print("Training pipeline completed successfully!")
    return model, history.history

def plot_and_save_curves(history, dataset_name, assets_dir):
    """
    Generates and saves accuracy and loss curves.
    """
    epochs_range = range(1, len(history['accuracy']) + 1)
    
    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history['accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, history['val_accuracy'], 'ro-', label='Validation Accuracy')
    plt.title(f'{dataset_name.upper()} - Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    acc_img_path = os.path.join(assets_dir, f'{dataset_name}_accuracy.png')
    plt.savefig(acc_img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy graph to {acc_img_path}")
    
    # Plot Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history['loss'], 'bo-', label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title(f'{dataset_name.upper()} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_img_path = os.path.join(assets_dir, f'{dataset_name}_loss.png')
    plt.savefig(loss_img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss graph to {loss_img_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN model for handwritten digit/character recognition.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'emnist'],
                        help='Dataset to use: mnist or emnist')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save the trained model')
    parser.add_argument('--assets_dir', type=str, default='static/assets',
                        help='Directory to save graphs')
    
    args = parser.parse_args()
    
    train(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_dir=args.model_dir,
        assets_dir=args.assets_dir
    )
