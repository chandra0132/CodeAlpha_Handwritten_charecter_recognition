import os
import argparse
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Ensure modules in src are importable
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_and_preprocess_dataset

def evaluate(model_path, dataset_name='mnist', assets_dir='static/assets'):
    """
    Evaluates a saved CNN model on the test dataset.
    Generates evaluation metrics and a confusion matrix heatmap.
    """
    os.makedirs(assets_dir, exist_ok=True)
    
    # 1. Load data
    _, _, x_test, y_test, class_names = load_and_preprocess_dataset(dataset_name)
    
    # 2. Load model
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}. Train it first using train.py.")
        
    model = tf.keras.models.load_model(model_path)
    
    # 3. Predict on test data
    print("Generating predictions on test set...")
    y_pred_probs = model.predict(x_test, batch_size=128, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # 4. Compute Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    print("\n" + "="*50)
    print(f"EVALUATION METRICS FOR {dataset_name.upper()} MODEL")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (Macro)")
    print(f"Recall:    {recall:.4f} (Macro)")
    print(f"F1-Score:  {f1:.4f} (Macro)")
    print("="*50)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save text report to a file
    report_path = os.path.join(os.path.dirname(model_path), f"{dataset_name}_evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"EVALUATION RESULTS FOR {dataset_name.upper()}\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f} (Macro)\n")
        f.write(f"Recall:    {recall:.4f} (Macro)\n")
        f.write(f"F1-Score:  {f1:.4f} (Macro)\n")
        f.write("="*50 + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Saved evaluation report text to {report_path}")
    
    # 5. Generate and Save Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10) if dataset_name == 'emnist' else (8, 6))
    
    # Choose heatmap parameters based on dataset size
    annot_flag = True if dataset_name == 'mnist' else False # Numbers inside cells might be too cluttered for EMNIST (47 classes)
    
    sns.heatmap(
        cm,
        annot=annot_flag,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    
    plt.title(f'Confusion Matrix Heatmap - {dataset_name.upper()}', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    cm_path = os.path.join(assets_dir, f"{dataset_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix heatmap to {cm_path}")
    
    # Save a JSON file with general summary metrics for Flask web UI
    summary_path = os.path.join(os.path.dirname(model_path), f"{dataset_name}_summary.json")
    summary = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "num_classes": len(class_names),
        "dataset_name": dataset_name
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Saved evaluation summary metrics to {summary_path}")
    
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained CNN model.')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the saved model (.keras file)')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'emnist'],
                        help='Dataset used to train the model: mnist or emnist')
    parser.add_argument('--assets_dir', type=str, default='static/assets',
                        help='Directory to save output graphs')
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model,
        dataset_name=args.dataset,
        assets_dir=args.assets_dir
    )
