# %%writefile /kaggle/working/main.py
import os
import glob
from data_preparation import prepare_yolo_dataset
from preprocessing import preprocess_image
from minutiae_extraction import get_terminations_bifurcations
from training import train_yolo_model, evaluate_yolo_model
from visualization import test_single_image

def main():
    # Prepare dataset
    print("Preparing YOLO dataset...")
    prepare_yolo_dataset(val_ratio=0.2)
    
    # Train YOLO model
    print("\nTraining YOLO model...")
    yolo_model = train_yolo_model("/kaggle/working/yolo_data")
    
    # Evaluate YOLO model
    print("\nEvaluating YOLO model...")
    y_true, y_pred = evaluate_yolo_model(yolo_model, "/kaggle/working/yolo_data")
    
    # Test on a single image
    print("\nTesting single image...")
    test_images = glob.glob(f"/kaggle/working/yolo_data/images/val/*.BMP")[:1]
    if test_images:
        test_single_image(yolo_model, test_images[0], preprocess_image, get_terminations_bifurcations)
    else:
        print("No validation images found for testing.")

if __name__ == "__main__":
    main()