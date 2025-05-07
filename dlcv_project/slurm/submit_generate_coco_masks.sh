#!/bin/bash
#SBATCH --job-name=generate_coco_masks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/home/pradeepd/logs/generate_coco_masks_%j.out
#SBATCH --error=/home/pradeepd/logs/generate_coco_masks_%j.err
#SBATCH --partition=debug

# Create logs directory
mkdir -p /home/pradeepd/logs

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Verify dependencies
echo "Verifying dependencies..."
conda list | grep pycocotools
pip show numpy
pip show psutil
pip show tqdm

# Check disk space
echo "Checking disk space..."
df -h /home/pradeepd
quota -s

# Change to scripts directory
cd /home/pradeepd/dlcv_project/scripts

# Verify input data
echo "Checking input data..."
if [ ! -f "/home/pradeepd/data/coco/annotations/instances_val2017.json" ]; then
    echo "Error: /home/pradeepd/data/coco/annotations/instances_val2017.json not found"
    exit 1
fi
if [ ! -d "/home/pradeepd/data/coco/val2017" ]; then
    echo "Error: /home/pradeepd/data/coco/val2017 not found"
    exit 1
fi

# Log system info
echo "Node: $(hostname)"
echo "CPUs allocated: $(nproc)"
nvidia-smi

# Run script
echo "Running generate_coco_masks.py..."
python generate_coco_masks.py

echo "Job completed"