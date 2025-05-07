import pickle
import numpy as np
import logging
import os
import psutil

# Placeholder for LangSplat import (replace with actual import)
# from langsplat import LangSplatModel

# Setup logging
logging.basicConfig(
    filename='/home/pradeepd/logs/langsplat_integration.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths
input_path = "/home/pradeepd/data/coco/3d_processed/rendering_ready.pkl"
output_dir = "/home/pradeepd/data/langsplat_output"
os.makedirs(output_dir, exist_ok=True)

# Validate disk space
def check_disk_space(path):
    disk = psutil.disk_usage(path)
    free_gb = disk.free / (1024 ** 3)
    logging.info(f"Disk space check for {path}: {free_gb:.2f} GB free")
    if free_gb < 3.0:
        logging.warning(f"Low disk space: {free_gb:.2f} GB free. Integration may fail.")
        print(f"Warning: Low disk space ({free_gb:.2f} GB free).")
    return free_gb

# Format data for LangSplat
def format_for_langsplat(entry):
    try:
        image_id = entry["image_id"]
        gaussians = entry["gaussians"]
        clip_embeddings = entry["clip_embeddings"]
        
        # Format Gaussians for LangSplat
        langsplat_gaussians = [
            {
                "mean": g["position"],
                "covariance": g["covariance"],
                "opacity": g["opacity"],
                "color": g["color"],
                "scale": g["scale"],
                "rotation": g["rotation"],
                "embedding": g.get("embedding", np.zeros(64))
            }
            for g in gaussians if g is not None
        ]
        
        langsplat_data = {
            "image_id": image_id,
            "gaussians": langsplat_gaussians,
            "clip_embeddings": clip_embeddings
        }
        
        # Save formatted data
        output_path = os.path.join(output_dir, f"langsplat_{image_id:012d}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(langsplat_data, f)
        logging.info(f"Formatted image ID {image_id} for LangSplat, saved to {output_path}")
        
        # Placeholder: Feed to LangSplat
        # langsplat_model = LangSplatModel()
        # langsplat_model.process_gaussians(langsplat_data["gaussians"])
        
        return langsplat_data
    except Exception as e:
        logging.error(f"Error formatting image ID {image_id}: {str(e)}")
        return None

# Main processing
logging.info("Starting LangSplat integration...")
check_disk_space("/home/pradeepd")

# Load data
try:
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    logging.info(f"Loaded {len(data)} entries from {input_path}")
except Exception as e:
    logging.error(f"Failed to load data: {str(e)}")
    raise

# Process for LangSplat
langsplat_results = []
for entry in tqdm(data, desc="Formatting for LangSplat"):
    result = format_for_langsplat(entry)
    if result:
        langsplat_results.append(result)

# Save all results
output_all_path = os.path.join(output_dir, "langsplat_all.pkl")
try:
    with open(output_all_path, "wb") as f:
        pickle.dump(langsplat_results, f)
    logging.info(f"Saved all LangSplat data to {output_all_path} with {len(langsplat_results)} entries")
    print(f"Saved all LangSplat data to {output_all_path}")
except Exception as e:
    logging.error(f"Failed to save LangSplat data: {str(e)}")
    raise

logging.info("LangSplat integration completed")
print("LangSplat integration completed")