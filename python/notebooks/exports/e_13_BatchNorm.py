# module automatically generated from 13_BatchNorm.ipynb
# to change this code, please edit the appropriate notebook and re-export, rather than editing this script directly

from exports.e_12_GeneralReLU import *

class BatchNorm(nn.Module):
    def __init__(self, n_inp, mntm=0.1, eps=1e-5):
        super().__init__()
        self.mntm, self.eps = mntm, eps
        
        self.gamma = nn.Parameter(torch.ones (n_inp, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(n_inp, 1, 1))

        self.register_buffer('means', torch.zeros(1, n_inp, 1, 1))
        self.register_buffer('vars',  torch.ones (1, n_inp, 1, 1))
        
    def update_stats(self, x):
        m = x.mean((0, 2, 3), keepdim=True)
        v = x.var ((0, 2, 3), keepdim=True)
        self.means.lerp_(m, self.mntm)
        self.vars .lerp_(v, self.mntm)
        return m, v
    
    def forward(self, x):
        if self.training:
            with torch.no_grad(): m,v = self.update_stats(x)
        else: m,v = self.means, self.vars
        x = (x - m) / (v + self.eps).sqrt()
        return x*self.gamma + self.beta

def conv_GReLU(n_inp, n_out, kernel_size=3, stride=2, btch_norm=True, **kwargs):
    layers = [nn.Conv2d(n_inp, n_out,
                       kernel_size,
                       padding=kernel_size//2,
                       stride=stride,
                       bias = not btch_norm),
              GReLU(**kwargs)]
    if btch_norm: layers.append(BatchNorm(n_out))
    
    return nn.Sequential(*layers)

def cnn_layers_GReLU(data_w, n_kernels, **kwargs):
    n_kernels = [1] + n_kernels
    
    return [
        conv_GReLU(n_kernels[i], n_kernels[i+1], 5 if i==0 else 3, **kwargs)
        for i in range(len(n_kernels)-1)
    ] + [nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(n_kernels[-1], data_w.n_out)]