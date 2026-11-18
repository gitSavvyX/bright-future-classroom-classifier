# data_split.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATA_DIR = 'data/balanced_dataset'
RANDOM_SEED = 42
TEST_SIZE = 0.2
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# --- Step 1: Pool ALL images from the source train/ and val/ folders ---
image_paths = []
labels = []

for source_split in ['train', 'val']:
    split_dir = os.path.join(DATA_DIR, source_split)
    if not os.path.isdir(split_dir):
        print(f"Skipping missing folder: {split_dir}")
        continue
    for class_name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for f in os.listdir(class_dir):
            if f.lower().endswith(IMAGE_EXTS):
                image_paths.append(os.path.join(class_dir, f))
                labels.append(class_name)

df = pd.DataFrame({'image_path': image_paths, 'label': labels})

if df.empty:
    raise SystemExit(f"No images found under {DATA_DIR}. Check the folder name.")

print(f"Total images pooled from source train+val: {len(df)}")
print("\nClass distribution (pooled):")
print(df['label'].value_counts(), "\n")

# --- Step 2: Reproducible, stratified split with fixed seed (required by brief) ---
train_df, val_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,      # fixed seed → reproducible
    stratify=df['label'],          # stratify → same class ratios in both splits
)

# --- Step 3: Save CSVs ---
train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)

# --- Step 4: Summary ---
print(f"New train rows: {len(train_df)}")
print(f"New val   rows: {len(val_df)}")

# --- Step 5: Sanity check ---
overlap = set(train_df['image_path']) & set(val_df['image_path'])
print(f"\nFile-path overlap between train and val: {len(overlap)}")
print("✅ Saved train.csv and val.csv")