import pickle
import numpy as np
from tqdm import tqdm
import logging
import os
import psutil

# Setup logging
logging.basicConfig(
    filename='/home/pradeepd/logs/prepare_rendering.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths
input_path = "/home/pradeepd/data/coco/3d_processed/optimized_embeddings.pkl"
output_path = "/home/pradeepd/data/coco/3d_processed/rendering_ready.pkl"
output_dir = "/home/pradeepd/data/coco/3d_processed/multi_res"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Validate disk space
def check_disk_space(path):
    disk = psutil.disk_usage(path)
    free_gb = disk.free / (1024 ** 3)
    logging.info(f"Disk space check for {path}: {free_gb:.2f} GB free")
    if free_gb < 3.0:
        logging.warning(f"Low disk space: {free_gb:.2f} GB free. Preparation may fail.")
        print(f"Warning: Low disk space ({free_gb:.2f} GB free).")
    return free_gb

# Initialize Gaussian attributes
def initialize_gaussian_attributes(gaussians):
    try:
        new_gaussians = []
        for g in gaussians:
            if g is None:
                new_gaussians.append(None)
                continue
            g_new = g.copy()
            g_new["opacity"] = min(max(g_new.get("opacity", 1.0), 0.1), 1.0)
            g_new["color"] = np.clip(g_new.get("color", np.random.rand(3)), 0, 1)
            g_new["scale"] = np.clip(g_new.get("scale", np.ones(3) * 0.1), 0.01, 1.0)
            g_new["rotation"] = g_new.get("rotation", np.array([1, 0, 0, 0]))  # Quaternion
            new_gaussians.append(g_new)
        return new_gaussians
    except Exception as e:
        logging.error(f"Error initializing Gaussian attributes: {str(e)}")
        return gaussians

# Generate multi-resolution data
def generate_multi_res_data(gaussians, image_id, resolutions=[128, 256, 512]):
    try:
        multi_res_data = {}
        valid_gaussians = [g for g in gaussians if g is not None]
        if not valid_gaussians:
            return {}
        
        for res in resolutions:
            scale_factor = res / 512.0
            res_gaussians = []
            for g in valid_gaussians:
                g_new = g.copy()
                g_new["scale"] = g["scale"] * scale_factor
                res_gaussians.append(g_new)
            multi_res_data[res] = res_gaussians
            res_path = os.path.join(output_dir, f"res_{res}_{image_id:012d}.pkl")
            with open(res_path, "wb") as f:
                pickle.dump(res_gaussians, f)
            logging.info(f"Saved resolution {res} data for image ID {image_id} to {res_path}")
        return multi_res_data
    except Exception as e:
        logging.error(f"Error generating multi-resolution data for image ID {image_id}: {str(e)}")
        return {}

# Main processing
logging.info("Starting rendering preparation...")
check_disk_space("/home/pradeepd")

# Load data
try:
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    logging.info(f"Loaded {len(data)} entries from {input_path}")
except Exception as e:
    logging.error(f"Failed to load data: {str(e)}")
    raise

# Prepare rendering data
rendering_data = []
for entry in tqdm(data, desc="Preparing rendering"):
    try:
        image_id = entry["image_id"]
        points_3d = entry["points_3d"]
        gaussians = entry["gaussians"]
        clip_embeddings = entry["clip_embeddings"]
        
        # Initialize Gaussian attributes
        new_gaussians = initialize_gaussian_attributes(gaussians)
        
        # Generate multi-resolution data
        multi_res_data = generate_multi_res_data(new_gaussians, image_id)
        
        rendering_data.append({
            "image_id": image_id,
            "points_3d": points_3d,
            "gaussians": new_gaussians,
            "clip_embeddings": clip_embeddings,
            "multi_res_data": multi_res_data
        })
    except Exception as e:
        logging.error(f"Error processing image ID {image_id}: {str(e)}")
        rendering_data.append(entry)
        continue

# Save rendering-ready data
try:
    with open(output_path, "wb") as f:
        pickle.dump(rendering_data, f)
    logging.info(f"Saved rendering-ready data to {output_path} with {len(rendering_data)} entries")
    print(f"Saved rendering-ready data to {output_path}")
except Exception as e:
    logging.error(f"Failed to save rendering-ready data: {str(e)}")
    raise

logging.info("Rendering preparation completed")
print("Rendering preparation completed")