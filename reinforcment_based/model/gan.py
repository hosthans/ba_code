import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_dim=100, dim=64):
        super(Generator, self).__init__()
        def conv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5,2, padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU()
        )

        self.l2_5 = nn.Sequential(
            conv_bn_relu(dim * 8, dim * 4),
            conv_bn_relu(dim * 4, dim * 2),
            conv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y
    

class DGWGAN(nn.Module):
    def __init__(self, in_dim=3, dim=64):
        super(DGWGAN, self).__init__()
        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2)
            )
        
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), 
            nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim*2),
            conv_ln_lrelu(dim*2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4)
        )

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y
    

class Generator_MNIST(nn.Module):
    def __init__(self, in_dim=100, dim=64):
        super(Generator_MNIST, self).__init__()
        def conv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2, padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU()
        )

        self.l2_5 = nn.Sequential(
            conv_bn_relu(dim * 8, dim * 4),
            conv_bn_relu(dim * 4, dim * 2),
            conv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 1, 5, 2, padding=2, output_padding=1),  # Änderung des Output-Kanals auf 1
            nn.Tanh()  # Verwendung von Tanh-Aktivierungsfunktion anstelle von Sigmoid
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y
    

class DGWGAN_MNIST(nn.Module):
    def __init__(self, in_dim=1, dim=64):
        super(DGWGAN_MNIST, self).__init__()
        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2)
            )
        
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), 
            nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim*2),
            conv_ln_lrelu(dim*2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4)
        )

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y