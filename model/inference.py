##
from tqdm import tqdm
from glob import glob
import numpy as np
import nibabel as nib
import os
##
import torch
from torch import nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
##
from model import MMMT
import losses as losses
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--t1", type=str, help="Path to the preprocessed T1 MRI")
parser.add_argument("--pet", type=str, help="Path to the preprocessed PET image")
parser.add_argument("--sd", type=str, help="Output directory where model outputs will be saved")
parser.add_argument("--use_gpu", action="store_true", help="Use GPU acceleration (default: cpu)")

args = parser.parse_args()
device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = MMMT(1, 8, 256, 256, 3, 1)
model.age_predictor.reg[0].bias = nn.Parameter(torch.tensor([0.], device = device, dtype = torch.float32))
model.label_predictor.classif[0].bias = nn.Parameter(torch.zeros(3, device = device, dtype = torch.float32).log())

pretrained_weights = "weights/pretrained_weights.pth"

checkpoints = torch.load(pretrained_weights, map_location='cpu')
model.load_state_dict(checkpoints)

model.to(device)
model.eval()

t1 = utils.decode(args.t1).unsqueeze(0).to(device) #add batch dim
pet = utils.decode(args.pet).unsqueeze(0).to(device) #add batch dim

with torch.no_grad():
  rec_t1, mu_zx_t1, logvar_zx_t1, rec_pet, mu_zx_pet, logvar_zx_pet, mu_ps, logvar_ps, z_ps = model.variational_model(t1, pet)
  logits_label = model.label_predictor(mu_ps)
  
os.makedirs(args.sd, exist_ok=True)

np.save(os.path.join(args.sd, "patientspace_classifier_pred.npy"), logits_label.softmax(dim = 1).detach().cpu().numpy())
np.save(os.path.join(args.sd, "patientspace_mu.npy"), mu_ps.detach().cpu().numpy())
np.save(os.path.join(args.sd, "patientspace_logvar.npy"), logvar_ps.detach().cpu().numpy())
