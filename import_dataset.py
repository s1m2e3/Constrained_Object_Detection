import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random


def get_labels_for_image(image_name, annotations_data):
    """Finds all annotations for a given image file name."""
    
    # Find the image ID from the image file name
    image_id = -1
    for img in annotations_data['images']:
        if img['file_name'] == image_name:
            image_id = img['id']
            break
    
    if image_id == -1:
        print(f"Image '{image_name}' not found in annotations file.")
        return []

    # Create a mapping from category ID to category name
    categories = {cat['id']: cat['name'] for cat in annotations_data['categories']}

    # Find all annotations for that image ID
    image_annotations = []
    for ann in annotations_data['annotations']:
        if ann['image_id'] == image_id:
            category_name = categories.get(ann['category_id'], 'unknown')
            image_annotations.append({
                'label': category_name,
                'bbox': ann['bbox'],  # [x, y, width, height]
                'label_id': ann['category_id']
            })
            
    return image_annotations

def batched_import(dataset_path="Self Driving Car.v3-fixed-small.coco",
                   image_directory='export',batch_size=16):

    # --- Main Script ---
    export_path = os.path.join(dataset_path, image_directory)

    # 1. Find the JSON annotation file
    annotation_file = None
    for filename in os.listdir(export_path):
        if filename.endswith('.json'):
            annotation_file = os.path.join(export_path, filename)
            break

    if annotation_file and os.path.exists(annotation_file):
        # 2. Load the annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # 3. Get all JPG filenames from the export directory
        all_jpg_files = [f for f in os.listdir(export_path) if f.lower().endswith('.jpg')]
        
        batches = [all_jpg_files[i:i + batch_size] for i in range(0, len(all_jpg_files), batch_size)]
        print(f"Created {len(batches)} batches of up to {batch_size} images each.")

        batch = random.choice(batches)
        print(f"\n--- Processing Batch ---")
        batch_image = []
        batch_label = []
        for filename in batch:
            image_path = os.path.join(export_path, filename)

            # Step 5a: Convert image to a NumPy array
            try:
                with Image.open(image_path) as img:
                    image_array = np.array(img)
                    batch_image.append(image_array)
            except Exception as e:
                print(f"  - Could not process image file: {e}")
                continue # Skip to the next file if image can't be opened
            # Step 5b: Get labels for the image
            labels = get_labels_for_image(filename, coco_data)
            if labels:
                bboxes = []
                for item in labels:
                    bboxes.append([item['label_id'],item['bbox']])
                batch_label.append(bboxes)
            else:
                print("  - No labels found for this image.")
                batch_label.append([]) # Append an empty list for images with no labels
        return batch_image,batch_label
    else:
        print("Could not find the COCO JSON annotation file in the 'export' directory.")
