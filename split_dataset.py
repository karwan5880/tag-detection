import os
import random
import shutil
from tqdm import tqdm

def split_dataset(source_images_dir, source_labels_dir, dest_dir, split_ratio=(0.8, 0.15, 0.05)):
    """
    Splits a dataset of images and labels into train, validation, and test sets.
    Handles images with and without corresponding labels (negative samples).

    Args:
        source_images_dir (str): Path to the folder containing all images.
        source_labels_dir (str): Path to the folder containing all labels.
        dest_dir (str): Path to the destination folder where the split dataset will be created.
        split_ratio (tuple): A tuple containing the ratios for (train, val, test). Must sum to 1.
    """
    # --- 1. Validation and Setup ---
    if sum(split_ratio) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    print("--- Starting Dataset Split ---")
    print(f"Source Images: {source_images_dir}")
    print(f"Source Labels: {source_labels_dir}")
    print(f"Destination: {dest_dir}")
    print(f"Split Ratio (Train/Val/Test): {split_ratio}")

    # Create destination directories
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        os.makedirs(os.path.join(dest_dir, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'labels', subset), exist_ok=True)
    print("\nCreated destination directory structure.")

    # --- 2. Identify Positive and Negative Samples ---
    all_image_files = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    positive_samples = []
    negative_samples = []

    for img_file in all_image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        if os.path.exists(os.path.join(source_labels_dir, label_file)):
            positive_samples.append(img_file)
        else:
            negative_samples.append(img_file)

    print(f"Found {len(positive_samples)} positive samples (with labels).")
    print(f"Found {len(negative_samples)} negative samples (without labels).")

    # Shuffle both lists randomly
    random.shuffle(positive_samples)
    random.shuffle(negative_samples)

    # --- 3. Function to process and copy files ---
    def process_and_copy(file_list, sample_type):
        train_split = int(len(file_list) * split_ratio[0])
        val_split = train_split + int(len(file_list) * split_ratio[1])

        splits = {
            'train': file_list[:train_split],
            'val': file_list[train_split:val_split],
            'test': file_list[val_split:]
        }
        
        copy_counts = {'train': 0, 'val': 0, 'test': 0}

        for subset, files in splits.items():
            for img_file in tqdm(files, desc=f"Copying {sample_type} to {subset}"):
                base_name = os.path.splitext(img_file)[0]
                
                # Copy image file
                src_img_path = os.path.join(source_images_dir, img_file)
                dest_img_path = os.path.join(dest_dir, 'images', subset, img_file)
                shutil.copy2(src_img_path, dest_img_path)

                # If it's a positive sample, copy its label file
                if sample_type == "positive":
                    label_file = f"{base_name}.txt"
                    src_label_path = os.path.join(source_labels_dir, label_file)
                    dest_label_path = os.path.join(dest_dir, 'labels', subset, label_file)
                    shutil.copy2(src_label_path, dest_label_path)
                
                copy_counts[subset] += 1
        return copy_counts

    # --- 4. Execute Splitting and Copying ---
    print("\n--- Processing Positive Samples ---")
    positive_counts = process_and_copy(positive_samples, "positive")

    print("\n--- Processing Negative Samples ---")
    negative_counts = process_and_copy(negative_samples, "negative")

    # --- 5. Final Summary ---
    print("\n--- Split Summary ---")
    for subset in subsets:
        total_images = positive_counts[subset] + negative_counts[subset]
        total_labels = positive_counts[subset] # Only positives have labels
        print(f"Subset: {subset.upper()}")
        print(f"  - Total Images: {total_images} ({positive_counts[subset]} positive, {negative_counts[subset]} negative)")
        print(f"  - Total Labels: {total_labels}")

    print("\nDataset splitting complete!")

if __name__ == '__main__':
    # --- Configuration ---
    # Your source folders containing all images and labels
    SOURCE_IMAGES_DIR = 'images/'
    SOURCE_LABELS_DIR = 'labels/'

    # The new folder where the structured dataset will be created
    DEST_DIR = 'my_custom_dataset/'

    # The split ratio for train, validation, and test sets
    SPLIT_RATIO = (0.8, 0.15, 0.05) # 80% train, 15% validation, 5% test
    # ---------------------

    split_dataset(SOURCE_IMAGES_DIR, SOURCE_LABELS_DIR, DEST_DIR, SPLIT_RATIO)