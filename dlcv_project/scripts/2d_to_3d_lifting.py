import os
import pickle
import numpy as np
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import logging
import psutil
import sys
sys.path.append('/home/pradeepd/dlcv_project/Code/midas')
from midas.model_loader import load_model

# Setup logging
logging.basicConfig(
    filename='/home/pradeepd/logs/lift_2d_to_3d.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths
input_pkl = '/home/pradeepd/data/coco/coco_processed.pkl'
image_dir = '/home/pradeepd/data/coco/val2017'
output_pkl = '/home/pradeepd/data/coco/3d_processed/3d_processed.pkl'
os.makedirs(os.path.dirname(output_pkl), exist_ok=True)

# Validate disk space
def check_disk_space(path):
    disk = psutil.disk_usage(path)
    free_gb = disk.free / (1024 ** 3)
    logging.info(f"Disk space check for {path}: {free_gb:.2f} GB free")
    if free_gb < 3.0:
        logging.warning(f"Low disk space: {free_gb:.2f} GB free. Lifting may fail.")
        print(f"Warning: Low disk space ({free_gb:.2f} GB free).")
    return free_gb

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load MiDaS
try:
    midas, midas_transform, midas_net_w, midas_net_h = load_model(
        device, model_path="/home/pradeepd/.cache/midas/dpt_large", model_type="dpt_large"
    )
    midas.eval()
    logging.info("Loaded MiDaS model")
except Exception as e:
    logging.error(f"Failed to load MiDaS: {str(e)}")
    raise

# Load CLIP
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    logging.info("Loaded CLIP model")
except Exception as e:
    logging.error(f"Failed to load CLIP: {str(e)}")
    raise

# Main processing
logging.info("Starting 2D-to-3D lifting...")
check_disk_space("/home/pradeepd")

# Load COCO data
try:
    with open(input_pkl, "rb") as f:
        coco_data = pickle.load(f)
    logging.info(f"Loaded {len(coco_data)} entries from {input_pkl}")
except Exception as e:
    logging.error(f"Failed to load COCO data: {str(e)}")
    raise

# Process images
three_d_data = []
for entry in tqdm(coco_data, desc="Lifting 2D to 3D"):
    try:
        image_id = entry['image_id']
        mask_path = entry['mask_path']
        category_ids = entry['category_ids']
        height, width = entry['height'], entry['width']
        
        # Load image
        image_path = os.path.join(image_dir, f"{image_id:012d}.jpg")
        if not os.path.exists(image_path):
            logging.warning(f"Image not found for ID {image_id}: {image_path}")
            continue
        image = Image.open(image_path).convert("RGB")
        
        # Compute depth with MiDaS
        input_image = midas_transform(image).to(device)
        with torch.no_grad():
            depth = midas(input_image).squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        
        # Compute CLIP embeddings
        inputs = clip_processor(images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            clip_embeddings = clip_model.get_image_features(**inputs).cpu().numpy()
        
        # Generate 3D points (simple projection)
        mask_array = np.load(mask_path)
        points_3d = []
        for mask in mask_array:
            y, x = np.where(mask)
            z = depth[y, x]
            points = np.stack([x, y, z], axis=1)
            points_3d.append(points)
        
        # Initialize sparse Gaussians (placeholder)
        gaussians = []
        for i, points in enumerate(points_3d):
            if len(points) == 0:
                gaussians.append(None)
                continue
            mean = np.mean(points, axis=0)
            covariance = np.cov(points.T) if len(points) > 1 else np.eye(3) * 0.01
            gaussians.append({
                'position': mean,
                'covariance': covariance,
                'category_id': category_ids[i]
            })
        
        three_d_data.append({
            'image_id': image_id,
            'points_3d': points_3d,
            'gaussians': gaussians,
            'clip_embeddings': clip_embeddings.tolist()
        })
        
        del input_image, depth, inputs, clip_embeddings
        torch.cuda.empty_cache()
        logging.info(f"Processed image ID {image_id}")
    except Exception as e:
        logging.error(f"Error processing image ID {image_id}: {str(e)}")
        continue

# Save 3D data
try:
    with open(output_pkl, "wb") as f:
        pickle.dump(three_d_data, f)
    logging.info(f"Saved 3D data to {output_pkl} with {len(three_d_data)} entries")
    print(f"Saved 3D data to {output_pkl}")
except Exception as e:
    logging.error(f"Failed to save 3D data: {str(e)}")
    raise

logging.info("2D-to-3D lifting completed")
print("2D-to-3D lifting completed")