# PatientSpace: A multimodal graph-based latent representation framework for modeling neurodegenerative disease heterogeneity

![Illustration of the PatientSpace.](https://github.com/dmanou/PatientSpace/blob/main/patientspace_figure.svg)

## Overview

Pytorch implementation of the paper "PatientSpace: A multimodal graph-based latent representation framework for modeling neurodegenerative disease heterogeneity"

## Installation instruction

### 1) Clone the repository

```bash
git clone https://github.com/dmanou/PatientSpace.git
cd PatientSpace
```
### 2) Create Conda environment (Python 3.9.16)

```bash
conda create -n patientspace python=3.9.16 -y
conda activate patientspace
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) PyTorch installation

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

## Load and store model pretrained weights

To run the inference script, you must first download the pre-trained weights.

```bash
mkdir -p model/weights
wget https://github.com/dmanou/PatientSpace/releases/download/v1.0.0/pretrained_weights.pth -O model/weights/pretrained_weights.pth
```
## Run inference on a single patient

After preprocessing the T1 MRI and PET images, run:

```bash
cd model
python -m inference \
    --t1 /path/to/preprocessed_t1.nii.gz \
    --pet /path/to/preprocessed_pet.nii.gz \
    --sd /path/to/output_directory
```

### Arguments

| Argument | Description |
|---|---|
| `--t1` | Path to the preprocessed T1 MRI |
| `--pet` | Path to the preprocessed PET image |
| `--sd` | Output directory where model outputs will be saved |
| `--use_gpu` | Use GPU acceleration (default: cpu) |

### Example

```bash
python -m inference \
    --t1 data/sub-001/t1.nii.gz \
    --pet data/sub-001/pet.nii.gz \
    --sd outputs/sub-001
    --use_gpu
```

### Outputs

The inference pipeline generates:

| File | Description |
|---|---|
| `patientspace_classifier_pred.npy` | Model prediction |
| `patientspace_mu.npy` | Subject latent space mean vector (`μ`) |
| `patientspace_logvar.npy` | Subject latent space log-variance vector (`log σ²`) |

All outputs are saved in:

```text
outputs/sub-001/
```

### Example output structure

```text
outputs/sub-001/
├── patientspace_classifier_pred.npy
├── patientspace_mu.npy
└── patientspace_logvar.npy
```
