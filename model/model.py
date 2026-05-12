import torch
from torch import nn
import torch.nn.functional as F

import architecture_blocks as blocks
import utils
import math

class MMMT(nn.Module):
    def __init__(self, in_channels, out_channels, zx_dim, z_ps_dim, y_dim, age_dim):
        super().__init__()

        ## multimodal encoders
        self.resnet_t1 = blocks.ConvEncoder(in_channels, out_channels)
        self.resnet_tep = blocks.ConvEncoder(in_channels, out_channels)
        self.latent_fusion = blocks.MultiViewFusion(in_channels, out_channels, z_ps_dim)

        ## Residual encoders
        self.residual_encoder_t1 = blocks.UnimodalEncoder(in_channels, out_channels, zx_dim)
        self.residual_encoder_tep = blocks.UnimodalEncoder(in_channels, out_channels, zx_dim)

        ## Prior Head

        self.encoder_prior = blocks.PriorEncoderBlock(z_ps_dim, y_dim)

        # ## Multi Task Head
        
        self.age_predictor = blocks.Regressor(z_ps_dim, age_dim)
        self.label_predictor = blocks.Classifier(z_ps_dim, y_dim)

        self.decoder_t1 = blocks.Decoder(in_channels, out_channels, zx_dim + z_ps_dim)
        self.decoder_tep = blocks.Decoder(in_channels, out_channels, zx_dim + z_ps_dim)

        ## Other parameters
        self.y_dim = y_dim
        self.age_dim = age_dim

    def reparametrization(self, mu, logvar):
        scale = torch.exp( 0.5 * logvar)
        return mu + torch.randn_like(scale) * scale
    
    def variational_model(self, x_t1, x_tep):
        ## Residual spaces
        mu_zx_t1, logvar_zx_t1 = self.residual_encoder_t1(x_t1)
        mu_zx_tep, logvar_zx_tep = self.residual_encoder_tep(x_tep)

        zx_t1 = self.reparametrization(mu_zx_t1, logvar_zx_t1)
        zx_tep = self.reparametrization(mu_zx_tep, logvar_zx_tep)

        ## Private and Shared spaces
        E_t1 = self.resnet_t1(x_t1)
        E_tep = self.resnet_tep(x_tep)
        mu_ps, logvar_ps = self.latent_fusion(E_t1, E_tep)

        ## Patient space
        z_ps = self.reparametrization(mu_ps, logvar_ps)

        ## Reconstruction
        z_t1 = torch.cat((zx_t1, z_ps), dim = 1)
        z_tep = torch.cat((zx_tep, z_ps), dim = 1)
        
        x_rec_t1 = self.decoder_t1(z_t1)
        x_rec_tep = self.decoder_tep(z_tep)

        return (x_rec_t1, mu_zx_t1, logvar_zx_t1, x_rec_tep, mu_zx_tep, logvar_zx_tep, mu_ps, logvar_ps, z_ps)
                
    def forward(self, x1_t1, x1_tep, x2_t1, x2_tep, label, age):
        x1_t1_rec, mu_zx1_t1, logvar_zx1_t1, x1_tep_rec, mu_zx1_tep, logvar_zx1_tep, mu_ps_x1, logvar_ps_x1, z_ps_x1 = self.variational_model(x1_t1, x1_tep)
        x2_t1_rec, mu_zx2_t1, logvar_zx2_t1, x2_tep_rec, mu_zx2_tep, logvar_zx2_tep, mu_ps_x2, logvar_ps_x2, z_ps_x2 = self.variational_model(x2_t1, x2_tep)

        logits_label = self.label_predictor(z_ps_x1)
        age_hat = self.age_predictor(z_ps_x1)

        ## encode prior

        label_onehot = F.one_hot(label, self.y_dim).float()

        prior_vector = label_onehot * age
        mu_pzy, logvar_pzy = self.encoder_prior(prior_vector)
        return (x1_t1_rec, mu_zx1_t1, logvar_zx1_t1, x1_tep_rec, mu_zx1_tep, logvar_zx1_tep, mu_ps_x1, logvar_ps_x1, logits_label, age_hat,
                x2_t1_rec, mu_zx2_t1, logvar_zx2_t1, x2_tep_rec, mu_zx2_tep, logvar_zx2_tep, mu_ps_x2, logvar_ps_x2,
                mu_pzy, logvar_pzy)
        
    def evaluate(self, x_t1, x_tep, label, age):
        with torch.no_grad():
            x_rec_t1, mu_zx_t1, logvar_zx_t1, x_rec_tep, mu_zx_tep, logvar_zx_tep, mu_ps, logvar_ps, z_ps = self.variational_model(x_t1, x_tep)

            ## Multi tasking
            
            logits_label = self.label_predictor(mu_ps)
            age_hat = self.age_predictor(mu_ps)

            ## encode prior

            label_onehot = F.one_hot(label, self.y_dim).float()

            prior_vector = label_onehot * age 
            mu_pzy, logvar_pzy = self.encoder_prior(prior_vector)

            return (x_rec_t1, mu_zx_t1, logvar_zx_t1, x_rec_tep, mu_zx_tep, logvar_zx_tep, mu_ps, logvar_ps, mu_pzy, logvar_pzy, logits_label, age_hat)
