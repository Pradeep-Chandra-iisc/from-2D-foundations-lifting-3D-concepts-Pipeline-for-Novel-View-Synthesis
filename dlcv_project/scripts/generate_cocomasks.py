import os
import pickle
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import logging
import psutil

# Setup logging to track processing and errors
# Adaptable for other datasets: Log file path can be modified for different datasets
logging.basicConfig(
    filename='/home/pradeepd/logs/generate_coco_masks.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define paths for COCO dataset
# Adaptable for other datasets: Replace with paths to your dataset's annotations and images
# Example for VOC: annotation_file = '/path/to/voc/annotations/voc2012.json'
# Example for Cityscapes: annotation_file = '/path/to/cityscapes/annotations/instances.json'
annotation_file = '/home/pradeepd/data/coco/annotations/instances_val2017.json'
image_dir = '/home/pradeepd/data/coco/val2017'
output_pkl = '/home/pradeepd/data/coco/coco_processed.pkl'
mask_dir = '/home/pradeepd/data/coco/masks'
os.makedirs(mask_dir, exist_ok=True)

# Validate disk space to ensure sufficient storage
# Adaptable for other datasets: Path can be changed to check space for other dataset directories
def check_disk_space(path):
    disk = psutil.disk_usage(path)
    free_gb = disk.free / (1024 ** 3)
    logging.info(f"Disk space check for {path}: {free_gb:.2f} GB free")
    if free_gb < 3.0:
        logging.warning(f"Low disk space: {free_gb:.2f} GB free. Mask generation may fail.")
        print(f"Warning: Low disk space ({free_gb:.2f} GB free).")
    return free_gb

# Main processing
logging.info("Starting mask generation...")
check_disk_space("/home/pradeepd")

# Initialize COCO API for annotations
# Adaptable for other datasets: Replace COCO with dataset-specific API or custom parser
# Example for VOC: Use ElementTree to parse XML annotations
# Example for Cityscapes: Use COCO API or custom JSON parser if format differs
try:
    coco = COCO(annotation_file)
    logging.info(f"Loaded annotations from {annotation_file}")
except Exception as e:
    logging.error(f"Failed to load annotations: {str(e)}")
    raise

# Get image IDs
# Adaptable for other datasets: Modify to extract image IDs from your dataset's annotation structure
image_ids = coco.getImgIds()
logging.info(f"Found {len(image_ids)} images")

# Process images to generate masks
coco_data = []
for img_id in tqdm(image_ids, desc="Generating masks"):
    try:
        # Load image info
        # Adaptable for other datasets: Adjust to extract image metadata (height, width) from your dataset
        img_info = coco.loadImgs(img_id)[0]
        height, width = img_info['height'], img_info['width']
        
        # Get annotations for the image
        # Adaptable for other datasets: Modify to retrieve segmentation data (e.g., polygons, RLE) from your dataset
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Generate masks for each annotation
        masks = []
        category_ids = []
        for ann in anns:
            if 'segmentation' in ann:
                # Use COCO's mask generation
                # Adaptable for other datasets: Replace with dataset-specific mask generation
                # Example for VOC: Parse polygon points and create binary mask using PIL or OpenCV
                # Example for Cityscapes: Handle instance-level polygons or semantic labels
                mask = coco.annToMask(ann)
                if mask.shape == (height, width):
                    masks.append(mask)
                    category_ids.append(ann['category_id'])
        
        if not masks:
            logging.warning(f"No valid masks for image ID {img_id}")
            continue
        
        # Stack masks into a single array
        # Adaptable for other datasets: Ensure mask array format matches downstream pipeline requirements
        mask_array = np.stack(masks, axis=0)  # Shape: (num_masks, height, width)
        
        # Save masks to disk
        # Adaptable for other datasets: Adjust mask file naming convention if needed
        mask_path = os.path.join(mask_dir, f'masks_{img_id:012d}.npy')
        np.save(mask_path, mask_array)
        
        # Store metadata for downstream processing
        # Adaptable for other datasets: Include dataset-specific metadata (e.g., class names, instance IDs)
        coco_data.append({
            'image_id': img_id,
            'mask_path': mask_path,
            'category_ids': category_ids,
            'height': height,
            'width': width
        })
        
        logging.info(f"Processed image ID {img_id}, saved masks to {mask_path}")
    except Exception as e:
        logging.error(f"Error processing image ID {img_id}: {str(e)}")
        continue

# Save metadata to pickle file
# Adaptable for other datasets: Modify output path and metadata structure as needed
try:
    with open(output_pkl, 'wb') as f:
        pickle.dump(coco_data, f)
    logging.info(f"Saved metadata to {output_pkl} with {len(coco_data)} entries")
    print(f"Saved metadata to {output_pkl}")
except Exception as e:
    logging.error(f"Failed to save metadata: {str(e)}")
    raise

logging.info("Mask generation completed")
print("Mask generation completed")