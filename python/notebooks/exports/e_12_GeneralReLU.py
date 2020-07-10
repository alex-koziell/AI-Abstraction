# module automatically generated from 12_GeneralReLU.ipynb
# to change this code, please edit the appropriate notebook and re-export, rather than editing this script directly

from exports.e_11_MoreStats import *

class GReLU(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak,self.sub,self.maxv = leak,sub,maxv
        
    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak else F.relu(x)
        if self.sub is not None: x.sub_(self.sub)
        if self.maxv is not None: x.clamp_max_(self.maxv)
        return x

    
def conv_GReLU(n_inp, n_out, kernel_size=3, stride=2, **kwargs):
    return nn.Sequential(
            nn.Conv2d(n_inp, n_out,
                      kernel_size,
                      padding=kernel_size//2,
                      stride=stride),
            GReLU(**kwargs))

    
def cnn_layers_GReLU(data_w, n_kernels, **kwargs):
    n_kernels = [1] + n_kernels
    
    return [
        conv_GReLU(n_kernels[i], n_kernels[i+1], 5 if i==0 else 3, **kwargs)
        for i in range(len(n_kernels)-1)
    ] + [nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(n_kernels[-1], data_w.n_out)]