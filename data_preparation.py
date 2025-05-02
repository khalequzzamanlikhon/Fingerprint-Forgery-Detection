# %%writefile /kaggle/working/data_preparation.py
import os
import random
import shutil
import glob
from tqdm import tqdm

# Define global paths
DATA_DIR = "/kaggle/input/socofing/SOCOFing"
YOLO_DATA_DIR = "/kaggle/working/yolo_data"
REAL_DIR = os.path.join(DATA_DIR, "Real")
ALTERED_DIR = os.path.join(DATA_DIR, "Altered")

def prepare_yolo_dataset(val_ratio=0.2):
    """
    Prepare dataset for YOLO by splitting Real and Altered images into train/val sets.
    """
    real_images = glob.glob(f"{REAL_DIR}/*.BMP")
    fake_images = glob.glob(f"{ALTERED_DIR}/*/*.BMP")
    
    if not real_images or not fake_images:
        raise ValueError("No images found in dataset. Check directory paths.")
    
    # Shuffle images
    random.shuffle(real_images)
    random.shuffle(fake_images)
    
    # Split into train/val
    val_real = real_images[:int(len(real_images) * val_ratio)]
    train_real = real_images[int(len(real_images) * val_ratio):]
    val_fake = fake_images[:int(len(fake_images) * val_ratio)]
    train_fake = fake_images[int(len(fake_images) * val_ratio):]
    
    def process_files(file_list, class_id, split):
        for src_path in tqdm(file_list, desc=f"Processing {split} class {class_id}"):
            file_name = os.path.basename(src_path)
            dst_img = f"{YOLO_DATA_DIR}/images/{split}/{file_name}"
            shutil.copy(src_path, dst_img)
            with open(f"{YOLO_DATA_DIR}/labels/{split}/{os.path.splitext(file_name)[0]}.txt", 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")  # Full image bounding box
    
    # Create YOLO directory structure
    os.makedirs(f"{YOLO_DATA_DIR}/images/train", exist_ok=True)
    os.makedirs(f"{YOLO_DATA_DIR}/labels/train", exist_ok=True)
    os.makedirs(f"{YOLO_DATA_DIR}/images/val", exist_ok=True)
    os.makedirs(f"{YOLO_DATA_DIR}/labels/val", exist_ok=True)
    
    process_files(train_real, 0, "train")
    process_files(val_real, 0, "val")
    process_files(train_fake, 1, "train")
    process_files(val_fake, 1, "val")
    
    # Create YOLO dataset.yaml
    yolo_config = f"""
path: {YOLO_DATA_DIR}
train: images/train
val: images/val
names:
  0: real
  1: fake
"""
    with open(f"{YOLO_DATA_DIR}/dataset.yaml", 'w') as f:
        f.write(yolo_config)
    
    print(f"Training samples: {len(os.listdir(f'{YOLO_DATA_DIR}/images/train'))}")
    print(f"Validation samples: {len(os.listdir(f'{YOLO_DATA_DIR}/images/val'))}")