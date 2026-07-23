import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib

def KL_DIV(q_mu, q_logvar, p_mu, p_logvar):
    return 0.5 * torch.sum(p_logvar - q_logvar - 1 + (q_mu - p_mu)**2 / (p_logvar.exp() + 1e-5) + q_logvar.exp() / (p_logvar.exp() + 1e-5), dim =-1)

def KL_DIV_normal(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - logvar.exp() - mu**2, dim =-1)

def log_prob(x, x_rec):
    return torch.sum((x.flatten(1) - x_rec.flatten(1)) ** 2, dim =-1)

def compute_fwhm_iso(shape, ref_shape=(440,440,159), ref_fwhm=6):
    if len(shape) > 3:
        shape = shape[:3]
    scale = np.mean(np.array(shape) / np.array(ref_shape))
    return ref_fwhm * scale

def decode_mri(path):
    x = nib.load(path).get_fdata()
    x = np.nan_to_num(x)
    mask = x > 0.
    mean = (x * mask).sum() / mask.sum()
    std = np.sqrt(((x * mask)**2).sum() / mask.sum() - mean**2)
    zscore_img = (x - mean * mask) / std
    zscore_img = zscore_img[10:170, 16:208, :160]
    zscore_img = np.expand_dims(zscore_img, axis = 0)
    zscore_img = torch.tensor(zscore_img, dtype = torch.float32)
    return zscore_img

def decode_pet(path, shape):
    x = nib.load(path)
    x_bis = nib.load(path).get_fdata()
    x_bis = np.nan_to_num(x_bis)
    mask = x_bis > 0.
    smooth_factor = compute_fwhm_iso(shape)
    smooth_x = processing.smooth_image(x, smooth_factor).get_fdata() * mask
    mean = (smooth_x * mask).sum() / mask.sum()
    std = np.sqrt(((smooth_x * mask)**2).sum() / mask.sum() - mean**2)
    zscore_img = (smooth_x - mean * mask) / std
    zscore_img = zscore_img[10:170, 16:208, :160]
    zscore_img = np.expand_dims(zscore_img, axis = 0)
    zscore_img = torch.tensor(zscore_img, dtype = torch.float32)
    return zscore_img
