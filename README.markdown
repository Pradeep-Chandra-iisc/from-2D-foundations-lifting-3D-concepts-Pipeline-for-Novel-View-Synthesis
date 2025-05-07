# 2D-to-3D Lifting Pipeline for Novel-View Synthesis

This repository implements a pipeline for lifting 2D images from the COCO dataset to 3D representations using sparse Gaussian splatting, optimized language embeddings, and LangSplat integration for novel-view synthesis. The pipeline processes COCO images to generate segmentation masks, 3D points, optimized embeddings, and rendering-ready data, with scripts designed to run on a cluster with A5000 GPUs.

## Repository Structure

```
dlcv_project/
├── README.md
├── scripts/
│   ├── generate_coco_masks.py
│   ├── lift_2d_to_3d.py
│   ├── optimize_embeddings.py
│   ├── prepare_rendering.py
│   ├── langsplat_integration.py
├── slurm/
│   ├── submit_generate_coco_masks.sh
│   ├── submit_lift_2d_to_3d.sh
│   ├── submit_optimize_embeddings.sh
│   ├── submit_prepare_rendering.sh
│   ├── submit_langsplat_integration.sh
├── requirements.txt
```

## Overview

The pipeline processes COCO 2017 validation images (~5,000 images) to:
1. Generate binary segmentation masks (`coco_processed.pkl`, `masks_<image_id>.npy`).
2. Lift 2D images to 3D using MiDaS depth estimation and CLIP embeddings (`3d_processed.pkl`).
3. Optimize CLIP embeddings with MLP and cross-attention (`optimized_embeddings.pkl`).
4. Prepare multi-resolution Gaussian data for rendering (`rendering_ready.pkl`, `multi_res/`).
5. Format data for LangSplat integration (`langsplat_<image_id>.pkl`, `langsplat_all.pkl`).

The pipeline is designed for a cluster with 6 nodes, A5000 GPUs, 48 CPUs, and ~4 GB free disk space in `/home/pradeepd` (60 GB quota). The `generate_coco_masks.py` script is adaptable for other datasets (e.g., Pascal VOC, Cityscapes) with modifications to annotation parsing and paths.

## Setup

### Prerequisites
- **Cluster**: SLURM-managed, A5000 GPUs, 48 CPUs, Python 3.10 in `myenv`.
- **Disk Space**: ~4 GB free in `/home/pradeepd`, 60 GB quota (COCO val2017 ~1.1 GB, annotations ~25 MB, outputs vary).
- **MiDaS Model**: `/home/pradeepd/.cache/midas/dpt_large`.
- **Tools**: `wget`, `unzip` installed on the cluster.

### Installation
1. **Clone Repository** (if applicable):
   ```bash
   cd /home/pradeepd
   mkdir dlcv_project
   cd dlcv_project
   ```

2. **Create Directory Structure**:
   ```bash
   mkdir -p scripts slurm
   ```

3. **Download COCO Data**:
   - Create directories for data and outputs:
     ```bash
     mkdir -p /home/pradeepd/data/coco/{val2017,annotations,masks,3d_processed/multi_res}
     mkdir -p /home/pradeepd/data/langsplat_output
     ```
   - Download COCO 2017 validation images (~1.1 GB):
     ```bash
     wget http://images.cocodataset.org/zips/val2017.zip -P /home/pradeepd/data/coco/
     unzip /home/pradeepd/data/coco/val2017.zip -d /home/pradeepd/data/coco/
     rm /home/pradeepd/data/coco/val2017.zip
     ```
   - Download COCO 2017 annotations (~25 MB):
     ```bash
     wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P /home/pradeepd/data/coco/
     unzip /home/pradeepd/data/coco/annotations_trainval2017.zip -d /home/pradeepd/data/coco/
     mv /home/pradeepd/data/coco/annotations/instances_val2017.json /home/pradeepd/data/coco/annotations/
     rm -rf /home/pradeepd/data/coco/annotations_trainval2017.zip /home/pradeepd/data/coco/annotations/*train*.json
     ```

4. **Save Files**:
   - Save scripts in `scripts/`, SLURM scripts in `slurm/`, `requirements.txt`, and this `README.md`.
   - Set permissions:
     ```bash
     chmod +x scripts/*.py
     chmod +x slurm/*.sh
     ls -l scripts/
     ls -l slurm/
     ```

5. **Activate Environment**:
   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate myenv
   ```

6. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Dependencies (from `requirements.txt`):
   ```
   pycocotools==2.0.7
   torch==2.0.1
   torchvision==0.15.2
   transformers==4.36.0
   timm==0.6.12
   psutil==5.9.5
   matplotlib==3.7.1
   scikit-learn==1.2.2
   tqdm==4.66.1
   numpy==1.24.3
   Pillow==9.5.0
   ```

7. **Verify Input Data**:
   ```bash
   ls -l /home/pradeepd/data/coco/annotations/instances_val2017.json
   ls -l /home/pradeepd/data/coco/val2017/ | wc -l
   ls -l /home/pradeepd/data/coco/3d_processed/optimized_gaussians.pkl
   ```

## Pipeline Steps

1. **Generate COCO Masks** (~1.4 hours):
   - **Script**: `scripts/generate_coco_masks.py`
   - **SLURM**: `slurm/submit_generate_coco_masks.sh`
   - **Description**: Generates binary segmentation masks using `pycocotools`. Adaptable for VOC, Cityscapes by modifying paths and parsers.
   - **Input**: `/home/pradeepd/data/coco/annotations/instances_val2017.json`, `/home/pradeepd/data/coco/val2017/`
   - **Output**: `/home/pradeepd/data/coco/coco_processed.pkl`, `/home/pradeepd/data/coco/masks/masks_<image_id>.npy`

2. **2D-to-3D Lifting** (~1.5 hours):
   - **Script**: `scripts/lift_2d_to_3d.py`
   - **SLURM**: `slurm/submit_lift_2d_to_3d.sh`
   - **Description**: Uses MiDaS for depth estimation, CLIP for embeddings, initializes sparse Gaussians.
   - **Input**: `/home/pradeepd/data/coco/coco_processed.pkl`, `/home/pradeepd/data/coco/val2017/`
   - **Output**: `/home/pradeepd/data/coco/3d_processed/3d_processed.pkl`

3. **Optimize Language Embeddings** (~2.7 minutes):
   - **Script**: `scripts/optimize_embeddings.py`
   - **SLURM**: `slurm/submit_optimize_embeddings.sh`
   - **Description**: Compresses 512D CLIP embeddings to 64D via MLP, applies cross-attention.
   - **Input**: `/home/pradeepd/data/coco/3d_processed/optimized_gaussians.pkl`
   - **Output**: `/home/pradeepd/data/coco/3d_processed/optimized_embeddings.pkl`

4. **Prepare for Novel-View Synthesis** (~8.6 minutes):
   - **Script**: `scripts/prepare_rendering.py`
   - **SLURM**: `slurm/submit_prepare_rendering.sh`
   - **Description**: Initializes Gaussian attributes, generates multi-resolution data (128, 256, 512).
   - **Input**: `/home/pradeepd/data/coco/3d_processed/optimized_embeddings.pkl`
   - **Output**: `/home/pradeepd/data/coco/3d_processed/rendering_ready.pkl`, `/home/pradeepd/data/coco/3d_processed/multi_res/`

5. **Integrate with LangSplat** (~1.9 minutes):
   - **Script**: `scripts/langsplat_integration.py`
   - **SLURM**: `slurm/submit_langsplat_integration.sh`
   - **Description**: Formats Gaussians for LangSplat compatibility.
   - **Input**: `/home/pradeepd/data/coco/3d_processed/rendering_ready.pkl`
   - **Output**: `/home/pradeepd/data/langsplat_output/langsplat_<image_id>.pkl`, `/home/pradeepd/data/langsplat_output/langsplat_all.pkl`

## Execution

### Run Individual Steps
```bash
cd /home/pradeepd/dlcv_project/slurm
sbatch submit_generate_coco_masks.sh
sbatch submit_lift_2d_to_3d.sh
sbatch submit_optimize_embeddings.sh
sbatch submit_prepare_rendering.sh
sbatch submit_langsplat_integration.sh
```

### Monitor Jobs
```bash
squeue -u $USER
tail -f /home/pradeepd/logs/generate_coco_masks_<JOBID>.out
tail -f /home/pradeepd/logs/lift_2d_to_3d_<JOBID>.out
tail -f /home/pradeepd/logs/optimize_embeddings_<JOBID>.out
tail -f /home/pradeepd/logs/prepare_rendering_<JOBID>.out
tail -f /home/pradeepd/logs/langsplat_integration_<JOBID>.out
```

## Runtime Analysis

Estimated runtimes for ~5,000 COCO val2017 images on 1 A5000 GPU, 8 CPUs, 32 GB RAM:
- **generate_coco_masks.py**: ~1.4 hours (CPU-bound, mask generation, I/O).
- **lift_2d_to_3d.py**: ~1.5 hours (GPU-bound, MiDaS/CLIP, image I/O).
- **optimize_embeddings.py**: ~2.7 minutes (GPU-bound, MLP/cross-attention).
- **prepare_rendering.py**: ~8.6 minutes (CPU-bound, multi-res I/O).
- **langsplat_integration.py**: ~1.9 minutes (CPU-bound, I/O for multiple files).

## Verification

### Check Outputs
```bash
ls -l /home/pradeepd/data/coco/coco_processed.pkl
ls -l /home/pradeepd/data/coco/masks/ | wc -l
ls -l /home/pradeepd/data/coco/3d_processed/3d_processed.pkl
ls -l /home/pradeepd/data/coco/3d_processed/optimized_embeddings.pkl
ls -l /home/pradeepd/data/coco/3d_processed/rendering_ready.pkl
ls -l /home/pradeepd/data/langsplat_output/langsplat_all.pkl
ls -l /home/pradeepd/data/coco/3d_processed/multi_res/ | wc -l
python -c "import pickle; print(len(pickle.load(open('/home/pradeepd/data/coco/coco_processed.pkl', 'rb'))))"
python -c "import pickle; print(len(pickle.load(open('/home/pradeepd/data/coco/3d_processed/3d_processed.pkl', 'rb'))))"
python -c "import pickle; print(len(pickle.load(open('/home/pradeepd/data/langsplat_output/langsplat_all.pkl', 'rb'))))"
```

## Troubleshooting

### Disk Space Issues
```bash
df -h /home/pradeepd
quota -s
```
If low (~4 GB free):
```bash
rm -rf /home/pradeepd/data/coco/test2017
find /home/pradeepd/data/coco/3d_processed/multi_res/ -type f -name "res_*_*.pkl" -mtime +1 -delete
```
Consider moving outputs to `/mnt/data` if writable.

### Dependency Issues
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv
conda list | grep -E 'pycocotools|torch|transformers|matplotlib|scikit-learn'
pip show psutil
pip show timm
pip show tqdm
pip show numpy
pip show Pillow
```
Reinstall if needed:
```bash
pip install -r requirements.txt
```

### Download Issues
If `wget` or `unzip` fails:
```bash
which wget
which unzip
```
Install if missing (may require admin):
```bash
sudo apt-get install wget unzip
```
Retry download commands. If COCO URLs fail, check mirrors or download manually from http://cocodataset.org/#download.

### GPU Memory Issues
```bash
nvidia-smi
```
For `lift_2d_to_3d.py`, reduce batch size or process fewer images if CUDA OOM occurs.

## Adapting for Other Datasets

The `generate_coco_masks.py` script is designed to be adaptable for other datasets (e.g., Pascal VOC, Cityscapes):
1. **Update Paths**:
   ```python
   annotation_file = '/path/to/voc/annotations/voc2012.xml'  # Example for VOC
   image_dir = '/path/to/voc/JPEGImages'
   ```
2. **Replace COCO API**:
   - For VOC, use `xml.etree.ElementTree` to parse XML.
   - For Cityscapes, use `pycocotools` or custom JSON parser.
   ```python
   import xml.etree.ElementTree as ET
   tree = ET.parse(annotation_file)
   ```
3. **Modify Mask Generation**:
   - Use `PIL` or `opencv-python` for polygon-to-mask conversion.
   ```python
   from PIL import Image, ImageDraw
   mask = Image.new('L', (width, height), 0)
   draw = ImageDraw.Draw(mask)
   ```
4. **Update Metadata**:
   - Adjust `coco_data` for dataset-specific fields (e.g., class names).

Update `submit_generate_coco_masks.sh` to check dataset-specific paths:
```bash
if [ ! -f "/path/to/voc/annotations/voc2012.xml" ]; then
    echo "Error: VOC annotations not found"
    exit 1
fi
```

## Notes

- **LangSplat**: Replace placeholder in `langsplat_integration.py` with actual LangSplat library.
- **Model Training**: MLP/cross-attention in `optimize_embeddings.py` uses random weights; train externally if needed.
- **Disk Management**: Monitor disk space closely due to ~4 GB free limit. COCO data requires ~1.2 GB; outputs may need more.
- **Cluster Access**: Request admin assistance for quota increase, `/scratch` access, or `wget`/`unzip` installation if needed.
- **MiDaS Model**: Ensure `/home/pradeepd/.cache/midas/dpt_large` exists or update path in `lift_2d_to_3d.py`.

For issues, share:
```bash
ls -ld /mnt/data
df -h /home/pradeepd
quota -s
ls -l /home/pradeepd/data/coco/annotations/instances_val2017.json
ls -l /home/pradeepd/data/coco/val2017/ | wc -l
python -c "import pickle; print(len(pickle.load(open('/home/pradeepd/data/coco/3d_processed/optimized_gaussians.pkl', 'rb'))))" 2>/dev/null || echo "optimized_gaussians.pkl not found"
```

</xArtifact>

---

### Instructions to Save and Use
1. **Save README**:
   ```bash
   cd /home/pradeepd/dlcv_project
   nano README.md
   ```
   Copy and paste the content, then save.

2. **Verify**:
   ```bash
   ls -l /home/pradeepd/dlcv_project/README.md
   cat /home/pradeepd/dlcv_project/README.md
   ```

3. **Set Up Repository**:
   - Create directories:
     ```bash
     mkdir -p scripts slurm
     ```
   - Save scripts, SLURM files, and `requirements.txt` (from previous responses).
   - Set permissions:
     ```bash
     chmod +x scripts/*.py
     chmod +x slurm/*.sh
     ls -l scripts/
     ls -l slurm/
     ```

4. **Download COCO Data**:
   - Run the download commands from the Setup section:
     ```bash
     mkdir -p /home/pradeepd/data/coco/{val2017,annotations,masks,3d_processed/multi_res}
     mkdir -p /home/pradeepd/data/langsplat_output
     wget http://images.cocodataset.org/zips/val2017.zip -P /home/pradeepd/data/coco/
     unzip /home/pradeepd/data/coco/val2017.zip -d /home/pradeepd/data/coco/
     rm /home/pradeepd/data/coco/val2017.zip
     wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P /home/pradeepd/data/coco/
     unzip /home/pradeepd/data/coco/annotations_trainval2017.zip -d /home/pradeepd/data/coco/
     mv /home/pradeepd/data/coco/annotations/instances_val2017.json /home/pradeepd/data/coco/annotations/
     rm -rf /home/pradeepd/data/coco/annotations_trainval2017.zip /home/pradeepd/data/coco/annotations/*train*.json
     ```
   - Verify:
     ```bash
     ls -l /home/pradeepd/data/coco/annotations/instances_val2017.json
     ls -l /home/pradeepd/data/coco/val2017/ | wc -l
     ```

5. **Install Dependencies**:
   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate myenv
   pip install -r requirements.txt
   ```

6. **Execute Pipeline**:
   ```bash
   cd /home/pradeepd/dlcv_project/slurm
   sbatch submit_generate_coco_masks.sh
   sbatch submit_lift_2d_to_3d.sh
   sbatch submit_optimize_embeddings.sh
   sbatch submit_prepare_rendering.sh
   sbatch submit_langsplat_integration.sh
   ```

---

### Notes
- **Content**: The `data/` folder is removed from the repository structure, and commands to download COCO `val2017/` (~1.1 GB) and `instances_val2017.json` (~25 MB) are added to the Setup section. Output paths remain absolute (e.g., `/home/pradeepd/data/coco/masks/`) to keep the repository clean.
- **Download Commands**: Use `wget` and `unzip` to fetch data from official COCO URLs, placing files in `/home/pradeepd/data/coco/`. The commands clean up zip files and irrelevant annotations to save space.
- **Adaptability**: Emphasizes `generate_coco_masks.py`’s flexibility for other datasets, with clear modification steps.
- **Dependencies**: References `requirements.txt` from the previous response, ensuring all necessary packages are listed.
- **Cluster**: Assumes SLURM, `myenv`, and sufficient permissions for `wget`/`unzip`. Outputs are written to `/home/pradeepd/data/`.
- **Runtime Analysis**: Focuses on the five specified scripts, excluding `validate_and_visualize.py`.
- **Disk Space**: COCO data requires ~1.2 GB; outputs (e.g., masks, Gaussians) may need more. The README includes disk space checks.

---

### Questions
- Share:
  ```bash
  ls -ld /mnt/data
  ls -ld /mnt/data/pradeepd
  df -h /home/pradeepd
  quota -s
  ls -l /home/pradeepd/data/coco/annotations/instances_val2017.json 2>/dev/null || echo "Annotations not found"
  ls -l /home/pradeepd/data/coco/val2017/ | wc -l 2>/dev/null || echo "Images not found"
  python -c "import pickle; print(len(pickle.load(open('/home/pradeepd/data/coco/3d_processed/optimized_gaussians.pkl', 'rb'))))" 2>/dev/null || echo "optimized_gaussians.pkl not found"
  which wget
  which unzip
  ```
- Is `/mnt/data` writable for outputs if `/home/pradeepd` fills up?
- Do you have `wget` and `unzip` installed? If not, can you request admin assistance?
- Do you need specific sections added (e.g., contributor info, license)?
- Can you confirm the path to `optimized_gaussians.pkl`?
- Do you need assistance with LangSplat integration or model training?

This updated README replaces the `data/` folder with download commands and aligns with your requirements. Save it, set up the repository, download the data, and share outputs or clarifications for further assistance!