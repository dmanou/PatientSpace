import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import utils

def compute_losses(x1_t1_rec, mu_zx1_t1, logvar_zx1_t1, x1_tep_rec, mu_zx1_tep, logvar_zx1_tep, mu_ps_x1, logvar_ps_x1, logits_label, age_hat,
                   x2_t1_rec, mu_zx2_t1, logvar_zx2_t1, x2_tep_rec, mu_zx2_tep, logvar_zx2_tep, mu_ps_x2, logvar_ps_x2,
                   mu_pzy, logvar_pzy,
                   x1_t1, x1_tep, x2_t1, x2_tep, label, age):
    
    RE_t1 = 0.5 * (torch.mean(utils.log_prob(x1_t1, x1_t1_rec), dim = 0) + torch.mean(utils.log_prob(x2_t1, x2_t1_rec), dim = 0))
    RE_tep = 0.5 * (torch.mean(utils.log_prob(x1_tep, x1_tep_rec), dim = 0) + torch.mean(utils.log_prob(x2_tep, x2_tep_rec), dim = 0))
    
    CE_loss = nn.CrossEntropyLoss(weight = class_weights)(logits_label, label)
    age_loss = nn.MSELoss()(age_hat, age)

    KLD_norm_t1 = 0.5 * (torch.mean(utils.KL_DIV_normal(mu_zx1_t1, logvar_zx1_t1), dim = 0) + torch.mean(utils.KL_DIV_normal(mu_zx2_t1, logvar_zx2_t1), dim = 0))
    KLD_norm_tep = 0.5 * (torch.mean(utils.KL_DIV_normal(mu_zx1_tep, logvar_zx1_tep), dim = 0) + torch.mean(utils.KL_DIV_normal(mu_zx2_tep, logvar_zx2_tep), dim = 0))
    KLD_prior = 0.5 * (torch.mean(utils.KL_DIV(mu_ps_x1, logvar_ps_x1, mu_pzy, logvar_pzy), dim = 0) + torch.mean(utils.KL_DIV(mu_ps_x2, logvar_ps_x2, mu_pzy, logvar_pzy), dim = 0))

    Contrastive_loss = 0.5 * torch.mean(utils.KL_DIV(mu_z1_ps, logvar_z1_ps, mu_z2_ps, logvar_z2_ps) + utils.KL_DIV(mu_z2_ps, logvar_z2_ps, mu_z1_ps, logvar_z1_ps), dim = 0)
    
    return (RE_t1, RE_tep, KLD_norm_t1, KLD_norm_tep, KLD_prior, Contrastive_loss, CE_loss, age_loss)

def evaluate_losses(X_t1_rec, mu_zX_t1, logvar_zX_t1, X_tep_rec, mu_zX_tep, logvar_zX_tep, mu_ps, logvar_ps, mu_pzy, logvar_pzy, logits_label, age_hat,
                    X_t1, X_tep, label, age):
    RE_t1 = torch.mean(utils.log_prob(X_t1, X_t1_rec), dim = 0)
    RE_tep = torch.mean(utils.log_prob(X_tep, X_tep_rec), dim = 0)
    CE_loss = nn.CrossEntropyLoss(weight = class_weights)(logits_label, label) 
    age_loss = nn.MSELoss()(age_hat, age)

    KLD_norm_t1 = torch.mean(utils.KL_DIV_normal(mu_zX_t1, logvar_zX_t1), dim = 0)
    KLD_norm_tep = torch.mean(utils.KL_DIV_normal(mu_zX_tep, logvar_zX_tep), dim = 0)

    KLD_prior = torch.mean(utils.KL_DIV(mu_ps, logvar_ps, mu_pzy, logvar_pzy), dim = 0)
    return RE_t1, RE_tep, KLD_norm_t1, KLD_norm_tep, KLD_prior, CE_loss, age_loss
