# PatientSpace

## Overview

Pytorch implementation of the paper "PatientSpace: An interpretable graph-based latent space for multimodal neuroimaging biomarker learning in Alzheimer’s Disease and Frontotemporal Dementia"

## Used modules

### Training modules

* Python 3.9.16
* Pytorch 2.0.1
* Nibabel 5.1.0
* MONAI 1.3.0

### Graph modules

* Python 3.10.15
* Nibabel 5.3.2
* NetworkX 3.4.2

## Run inference on a single patient

After preprocessing the T1 MRI and PET images, run:

```bash
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

### Example

```bash
python -m inference \
    --t1 data/sub-001/t1.nii.gz \
    --pet data/sub-001/pet.nii.gz \
    --sd outputs/sub-001
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
