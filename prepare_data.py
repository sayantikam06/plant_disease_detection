# prepare_data.py
# STEP 3 & 4 - Extract and Split Dataset

import zipfile
import os
import shutil
import random

# ─── CHANGE THIS to your ZIP file location ───
zip_path = r"C:\Users\sayan\Downloads\archive.zip"

# ─── This will be inside your project folder ───
extract_path = r"C:\Users\sayan\Downloads\archive"
output_dir = "dataset_split"
train_dir = os.path.join(output_dir, "train")
val_dir   = os.path.join(output_dir, "val")

# STEP 3 - Extract
#print("Extracting dataset...")
#with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    #zip_ref.extractall(extract_path)
#print("Extraction complete!")

# STEP 4 - Split 80% train / 20% val
print("Splitting dataset...")
source_dir = os.path.join(extract_path, "plantvillage dataset", "color")
random.seed(42)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)
    split_point = int(len(images) * 0.8)

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    for img in images[:split_point]:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(train_dir, class_name, img))

    for img in images[split_point:]:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(val_dir, class_name, img))

print("Done! Dataset split into train and val folders.")