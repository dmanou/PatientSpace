import torch
from torch import nn
import torch.nn.functional as F

## ResNet Block

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels = out_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_channels, out_channels = out_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(8, out_channels),
        )

        self.act = nn.ReLU()

    def forward(self, x):
        b = self.bottleneck(x)
        return self.act(x + b)
    
class DonwsampleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DonwsampleResidualBlock, self).__init__()

        self.ds = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels = out_channels, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU(),
        )
        
        self.resblock = ResidualBlock(out_channels, out_channels)
        
    def forward(self, x):
        downsample = self.ds(x)
        return self.resblock(downsample)

class MMTM(nn.Module):
    def __init__(self, in_channels):
        super(MMTM, self).__init__()

        self.in_channels = in_channels

        self.S_t1 = nn.Sequential(
            nn.AdaptiveAvgPool3d((1)),
            nn.Flatten(),
        )

        self.S_tep = nn.Sequential(
            nn.AdaptiveAvgPool3d((1)),
            nn.Flatten(),
        )

        self.lin = nn.Linear(in_channels * 2, in_channels // 4)
        self.t1_lin = nn.Linear(in_channels // 4, in_channels)
        self.tep_lin = nn.Linear(in_channels // 4, in_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, h_t1, h_tep):
        s_t1 = self.S_t1(h_t1)
        s_tep = self.S_tep(h_tep)

        S = torch.cat((s_t1, s_tep), dim = 1)
        Z = self.lin(S)

        E_t1 = self.t1_lin(Z)
        E_tep = self.tep_lin(Z)

        A_t1 = (2 * self.sigmoid(E_t1)).view(-1, self.in_channels, 1, 1, 1)
        A_tep = (2 * self.sigmoid(E_tep)).view(-1, self.in_channels, 1, 1, 1)

        return A_t1 * h_t1, A_tep * h_tep
       

class UpSampleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleResidualBlock, self).__init__()

        self.us = nn.Sequential(
            nn.Upsample(scale_factor=2, mode = 'trilinear'),
            nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = 1, padding = 0),
        )
        
        self.resblock = ResidualBlock(out_channels, out_channels)
        
    def forward(self, x):
        downsample = self.us(x)
        return self.resblock(downsample)
    
class UnimodalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim):
        super(UnimodalEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 5, stride = 1, padding = 2),
            ResidualBlock(out_channels, out_channels),
        )
        
        self.conv2 = nn.Sequential(
            DonwsampleResidualBlock(out_channels, out_channels * 2), #80, 96, 80
        )
        
        self.conv3 = nn.Sequential(
            DonwsampleResidualBlock(out_channels * 2, out_channels * 4),
        )

        self.conv4 = nn.Sequential(
            DonwsampleResidualBlock(out_channels * 4, out_channels * 8), #20, 24, 20
        )

        self.conv5 = nn.Sequential(
            DonwsampleResidualBlock(out_channels * 8, out_channels * 16), #10, 12, 10
        )

        self.conv6 = nn.Sequential(
            DonwsampleResidualBlock(out_channels * 16, out_channels * 32), #5, 6, 5
            ResidualBlock(out_channels * 32, out_channels * 32),
        )

        self.mu = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5 * 6 * 5 * out_channels * 32, z_dim),
        )
        self.logvar = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5 * 6 * 5 * out_channels * 32, z_dim),
        )
    
    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        embeddings = self.conv6(h)
        return self.mu(embeddings), self.logvar(embeddings)
    
class ConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvEncoder, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 5, stride = 1, padding = 2),
            ResidualBlock(out_channels, out_channels),
        )
        
        self.conv2 = nn.Sequential(
            DonwsampleResidualBlock(out_channels, out_channels * 2), #80, 96, 80
        )
        
        self.conv3 = nn.Sequential(
            DonwsampleResidualBlock(out_channels * 2, out_channels * 4), #40, 48, 40
        )

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        fe = self.conv3(h)
        return fe

class MMTMEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MMTMEncoder, self).__init__()

        self.mmtm1 = MMTM(out_channels * 4)
        self.conv1_t1 = DonwsampleResidualBlock(out_channels * 4, out_channels * 8)
        self.conv1_tep = DonwsampleResidualBlock(out_channels * 4, out_channels * 8)

        self.mmtm2 = MMTM(out_channels * 8)
        self.conv2_t1 = DonwsampleResidualBlock(out_channels * 8, out_channels * 16)
        self.conv2_tep = DonwsampleResidualBlock(out_channels * 8, out_channels * 16)

        self.mmtm3 = MMTM(out_channels * 16)
        self.conv3_t1 = DonwsampleResidualBlock(out_channels * 16, out_channels * 32)
        self.conv3_tep = DonwsampleResidualBlock(out_channels * 16, out_channels * 32)

    def forward(self, t1, tep):
        h1_t1, h1_tep = self.mmtm1(t1, tep)
        h1_t1 = self.conv1_t1(h1_t1)
        h1_tep = self.conv1_tep(h1_tep)

        h2_t1, h2_tep = self.mmtm2(h1_t1, h1_tep)
        h2_t1 = self.conv2_t1(h2_t1)
        h2_tep = self.conv2_tep(h2_tep)

        h3_t1, h3_tep = self.mmtm3(h2_t1, h2_tep)
        h3_t1 = self.conv3_t1(h3_t1)
        h3_tep = self.conv3_tep(h3_tep)
        return h3_t1, h3_tep
        

class FusionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionResidualBlock, self).__init__()

        self.channel_pooling = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels = out_channels, kernel_size=3, stride = 1, padding = 1),
            nn.GroupNorm(8, out_channels),
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels = out_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_channels, out_channels = out_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(8, out_channels),
        )

        self.act = nn.ReLU()
        
    def forward(self, x):
        ch_pool = self.channel_pooling(x)
        b = self.bottleneck(x)
        return self.act(ch_pool + b)

class MultiViewFusion(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim):
        super(MultiViewFusion, self).__init__()

        self.multimodal_encoders = MMTMEncoder(in_channels, out_channels)
        
        self.fusion_extraction = FusionResidualBlock(out_channels * 32 * 2, out_channels * 32)
        self.mu = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5 * 6 * 5 * out_channels * 32, z_dim)
        )

        self.logvar = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5 * 6 * 5 * out_channels * 32, z_dim)
        )

    def forward(self, t1, tep):
        h1, h2 = self.multimodal_encoders(t1, tep)
        H = torch.cat((h1, h2), dim = 1)
        embeddings = self.fusion_extraction(H)
        return self.mu(embeddings), self.logvar(embeddings)
    
class Classifier(nn.Module):
    def __init__(self, z_dim, target_dim):
        super(Classifier, self).__init__()
        self.classif = nn.Sequential(
            nn.Dropout(),
            nn.Linear(z_dim, target_dim)
        )

    def forward(self, z):
        logits = self.classif(z)
        return logits

class Regressor(nn.Module):
    def __init__(self, z_dim, target_dim):
        super(Regressor, self).__init__()
        self.reg = nn.Sequential(
            nn.Linear(z_dim, target_dim)
        )

    def forward(self, z):
        logits = self.reg(z)
        return logits
        
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.fc_decode = nn.Linear(z_dim, 5 * 6 * 5 * out_channels * 32)
        
        self.upsampling = nn.Upsample(scale_factor=2, mode = 'trilinear')

        self.upconv1 = nn.Sequential(
            UpSampleResidualBlock(out_channels * 32, out_channels * 16), #10,12,10
        )

        self.upconv2 = nn.Sequential(
            UpSampleResidualBlock(out_channels * 16, out_channels * 8), #20,24,20
        )

        self.upconv3 = nn.Sequential(
            UpSampleResidualBlock(out_channels * 8, out_channels * 4), #40,48,40
        )
        self.upconv4 = nn.Sequential(
            UpSampleResidualBlock(out_channels * 4, out_channels * 2), #80,96,80
        )
        self.upconv5 = nn.Sequential(
            UpSampleResidualBlock(out_channels * 2, out_channels), #160,192,160
            nn.Conv3d(in_channels = out_channels, out_channels = in_channels, kernel_size = 1, stride = 1, padding = 0),
        )

    def forward(self, z):
        h = self.fc_decode(z)
        h = h.view(-1,self.out_channels * 32, 5, 6, 5)
        h = self.upconv1(h)
        h = self.upconv2(h)
        h = self.upconv3(h)
        h = self.upconv4(h)
        x_rec = self.upconv5(h)
        return x_rec

class PriorEncoderBlock(nn.Module):
    def __init__(self, z_dim, label_dim):
        super(PriorEncoderBlock, self).__init__()

        self.mu_prior = nn.Sequential(
            nn.Linear(label_dim, z_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(z_dim // 2, z_dim),
            nn.LeakyReLU(),
            nn.Linear(z_dim, z_dim),
        )

        self.logvar_prior = nn.Sequential(
            nn.Linear(label_dim, z_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(z_dim // 2, z_dim),
            nn.LeakyReLU(),
            nn.Linear(z_dim, z_dim),
        )

    def forward(self, y):
        return self.mu_prior(y), self.logvar_prior(y)
