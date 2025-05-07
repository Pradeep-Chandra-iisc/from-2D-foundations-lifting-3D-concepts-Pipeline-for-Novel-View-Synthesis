import pickle
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
import os
import psutil

# Setup logging
logging.basicConfig(
    filename='/home/pradeepd/logs/optimize_embeddings.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths
input_path = "/home/pradeepd/data/coco/3d_processed/optimized_gaussians.pkl"
output_path = "/home/pradeepd/data/coco/3d_processed/optimized_embeddings.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Validate disk space
def check_disk_space(path):
    disk = psutil.disk_usage(path)
    free_gb = disk.free / (1024 ** 3)
    logging.info(f"Disk space check for {path}: {free_gb:.2f} GB free")
    if free_gb < 3.0:
        logging.warning(f"Low disk space: {free_gb:.2f} GB free. Optimization may fail.")
        print(f"Warning: Low disk space ({free_gb:.2f} GB free).")
    return free_gb

# Lightweight MLP for embedding compression
class EmbeddingCompressor(nn.Module):
    def __init__(self, input_dim=512, output_dim=64):
        super(EmbeddingCompressor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.mlp(x)

# Cross-attention mechanism
class CrossAttention(nn.Module):
    def __init__(self, embed_dim=64):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = 1.0 / (embed_dim ** 0.5)
    
    def forward(self, gaussian_embeds, global_embed):
        q = self.query(gaussian_embeds)
        k = self.key(global_embed)
        v = self.value(global_embed)
        attn = torch.softmax(q @ k.T * self.scale, dim=-1)
        return attn @ v

# Main processing
logging.info("Starting embedding optimization...")
check_disk_space("/home/pradeepd")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
try:
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    logging.info(f"Loaded {len(data)} entries from {input_path}")
except Exception as e:
    logging.error(f"Failed to load data: {str(e)}")
    raise

# Initialize models
compressor = EmbeddingCompressor(input_dim=512, output_dim=64).to(device).eval()
cross_attention = CrossAttention(embed_dim=64).to(device).eval()

# Note: Using random weights for simplicity; train models in practice
logging.info("Using untrained compressor and attention models (placeholder)")

optimized_data = []
for entry in tqdm(data, desc="Optimizing embeddings"):
    try:
        image_id = entry["image_id"]
        points_3d = entry["points_3d"]
        gaussians = entry["gaussians"]
        clip_embeddings = entry["clip_embeddings"]
        
        if not clip_embeddings or not gaussians:
            logging.warning(f"Skipping image ID {image_id}: Missing embeddings or Gaussians")
            optimized_data.append(entry)
            continue
        
        # Compress embeddings
        embeddings_tensor = torch.tensor(np.array(clip_embeddings), dtype=torch.float32).to(device)
        with torch.no_grad():
            compressed_embeds = compressor(embeddings_tensor).cpu().numpy()
        
        # Compute global embedding (average of compressed embeddings)
        global_embed = torch.mean(
            torch.tensor(compressed_embeds, dtype=torch.float32).to(device), 
            dim=0, 
            keepdim=True
        )
        
        # Assign embeddings to Gaussians via cross-attention
        valid_gaussians = [g for g in gaussians if g is not None]
        if not valid_gaussians:
            logging.warning(f"Skipping image ID {image_id}: No valid Gaussians")
            optimized_data.append(entry)
            continue
        
        gaussian_embeds = torch.tensor(
            compressed_embeds[:len(valid_gaussians)], 
            dtype=torch.float32
        ).to(device)
        with torch.no_grad():
            final_embeds = cross_attention(gaussian_embeds, global_embed).cpu().numpy()
        
        # Update Gaussians with embeddings
        new_gaussians = []
        embed_idx = 0
        for g in gaussians:
            if g is None:
                new_gaussians.append(None)
                continue
            g_new = g.copy()
            if embed_idx < len(final_embeds):
                g_new["embedding"] = final_embeds[embed_idx]
                embed_idx += 1
            else:
                g_new["embedding"] = np.zeros(64)
            new_gaussians.append(g_new)
        
        optimized_data.append({
            "image_id": image_id,
            "points_3d": points_3d,
            "gaussians": new_gaussians,
            "clip_embeddings": compressed_embeds.tolist()
        })
        
        del embeddings_tensor, compressed_embeds, global_embed, gaussian_embeds, final_embeds
        torch.cuda.empty_cache()
    except Exception as e:
        logging.error(f"Error processing image ID {image_id}: {str(e)}")
        optimized_data.append(entry)
        continue

# Save optimized data
try:
    with open(output_path, "wb") as f:
        pickle.dump(optimized_data, f)
    logging.info(f"Saved optimized data to {output_path} with {len(optimized_data)} entries")
    print(f"Saved optimized data to {output_path}")
except Exception as e:
    logging.error(f"Failed to save optimized data: {str(e)}")
    raise

logging.info("Embedding optimization completed")
print("Embedding optimization completed")