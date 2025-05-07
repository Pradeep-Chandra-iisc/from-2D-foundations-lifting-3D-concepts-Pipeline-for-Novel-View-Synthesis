#!/bin/bash
#SBATCH --job-name=optimize_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/pradeepd/logs/optimize_embeddings_%j.out
#SBATCH --error=/home/pradeepd/logs/optimize_embeddings_%j.err
#SBATCH --partition=debug

# Create logs directory
mkdir -p /home/pradeepd/logs

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Verify dependencies
echo "Verifying dependencies..."
conda list | grep -E 'pytorch|torchvision'
pip show psutil

# Check disk space
echo "Checking disk space..."
df -h /home/pradeepd
quota -s

# Change to scripts directory
cd /home/pradeepd/dlcv_project/scripts

# Verify input data
echo "Checking input data..."
if [ ! -f "/home/pradeepd/data/coco/3d_processed/optimized_gaussians.pkl" ]; then
    echo "Error: /home/pradeepd/data/coco/3d_processed/optimized_gaussians.pkl not found"
    exit 1
fi

# Log system info
echo "Node: $(hostname)"
echo "CPUs allocated: $(nproc)"
nvidia-smi

# Run script
echo "Running optimize_embeddings.py..."
python optimize_embeddings.py

echo "Job completed"