# PatientSpace

## Overview

Pytorch implementation of the paper "PatientSpace: An interpretable graph-based latent space for multimodal neuroimaging biomarker learning in Alzheimer’s Disease and Frontotemporal Dementia"

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

## Run inference on a single patient

After preprocessing the T1 MRI and PET images, run:

```bash
python -m inference \
    --t1 /path/to/preprocessed_t1.nii.gz \
    --pet /path/to/preprocessed_pet.nii.gz \
    --sd /path/to/output_directory
    --device {gpu / cpu}
```

### Arguments

| Argument | Description |
|---|---|
| `--t1` | Path to the preprocessed T1 MRI |
| `--pet` | Path to the preprocessed PET image |
| `--sd` | Output directory where model outputs will be saved |
| `--device` | Device to use gpu or cpu |

### Example

```bash
python -m inference \
    --t1 data/sub-001/t1.nii.gz \
    --pet data/sub-001/pet.nii.gz \
    --sd outputs/sub-001
    --device gpu
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
