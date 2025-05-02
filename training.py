from ultralytics import YOLO
import numpy as np
import torch
import os
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def train_yolo_model(yolo_data_dir):
    """
    Train YOLOv11 model for real vs. fake classification.
    """
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"GPU Names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    
    # Set and verify working directory
    os.chdir(yolo_data_dir)
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    yaml_path = os.path.join(yolo_data_dir, 'dataset.yaml')
    print(f"dataset.yaml path: {yaml_path}")
    print(f"dataset.yaml exists: {os.path.exists(yaml_path)}")
    
    # Verify train and val directories
    train_dir = os.path.join(yolo_data_dir, 'images', 'train')
    val_dir = os.path.join(yolo_data_dir, 'images', 'val')
    print(f"Train directory exists: {os.path.isdir(train_dir)}")
    print(f"Validation directory exists: {os.path.isdir(val_dir)}")
    
    # Read dataset.yaml for debugging
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            print("Content of dataset.yaml:")
            print(f.read())
    
    model = YOLO('yolo11s.pt')  
    training_params = {
        'data': yaml_path,  
        'epochs': 15,
        'imgsz': 640,
        'batch': 16,
        'device': 'cuda',
        'optimizer': 'Adam',
        'seed': 42
    }
    results = model.train(**training_params)
    return model

def evaluate_yolo_model(model, data_path):
    """
    Evaluate YOLO model on validation set.
    """
    y_true = []
    y_pred = []
    confidences = []
    
    val_images = [f for f in os.listdir(f"{data_path}/images/val") 
                  if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))]
    
    for img_name in tqdm(val_images, desc="Evaluating"):
        label_path = f"{data_path}/labels/val/{os.path.splitext(img_name)[0]}.txt"
        with open(label_path, 'r') as f:
            true_class = int(f.read().strip().split()[0])
        y_true.append(true_class)
        
        results = model.predict(f"{data_path}/images/val/{img_name}", verbose=False)
        if len(results[0].boxes) > 0:
            pred_class = int(results[0].boxes.cls[0].item())
            confidence = float(results[0].boxes.conf[0].item())
            y_pred.append(pred_class)
            confidences.append(confidence)
        else:
            y_pred.append(-1)
            confidences.append(0.0)
    
    valid_idx = [i for i, pred in enumerate(y_pred) if pred != -1]
    y_true = np.array(y_true)[valid_idx]
    y_pred = np.array(y_pred)[valid_idx]
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['real', 'fake']))
    
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Weighted F1 Score: {f1:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Real', 'Predicted Fake'],
                yticklabels=['Actual Real', 'Actual Fake'])
    plt.title(f'Confusion Matrix (F1 = {f1:.2f})')
    plt.show()
    
    return y_true, y_pred