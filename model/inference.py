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
parser.add_argument("--t1", type=str)
parser.add_argument("--pet", type=str)
parser.add_argument("--sd", type=str)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = MMMT(1, 8, 256, 256, 3, 1)

pretrained_weights = "model/weights/pretrained_weights.pth"

checkpoints = torch.load(pretrained_weights, map_location='cpu')
model.load_state_dict(checkpoints)

model.to(device)
model.eval()

t1 = utils.decode(args.t1).unsqueeze(0).to(device) #add batch dim
pet = utils.decode(args.pet).unsqueeze(0).to(device) #add batch dim

with torch.no_grad():
  rec_t1, mu_zx_t1, logvar_zx_t1, rec_pet, mu_zx_pet, logvar_zx_pet, mu_ps, logvar_ps, z_ps = model.variational_model(t1, pet)
  logits_label = model.label_predictor(mu_ps)

np.save(args.sd + "patientspace_classifier_pred.npy", logits_label.softmax(dim = 1).detach().cpu().numpy())
np.save(args.sd + "patientspace_mu.npy", mu_ps.detach().cpu().numpy())
np.save(args.sd + "patientspace_logvar.npy", logvar_ps.detach().cpu().numpy())


