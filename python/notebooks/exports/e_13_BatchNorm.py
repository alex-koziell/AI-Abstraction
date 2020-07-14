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

class RunningBatchNorm(nn.Module):
    def __init__(self, n_inp, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom, self.eps = mom, eps
        self.gamma = nn.Parameter(torch.ones (n_inp, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(n_inp, 1, 1))
        self.register_buffer('sums', torch.zeros(1,n_inp,1,1))
        self.register_buffer('sqrs', torch.zeros(1,n_inp,1,1))
        self.register_buffer('batch', torch.tensor(0.))
        self.register_buffer('count', torch.tensor(0.))
        self.register_buffer('step',  torch.tensor(0.))
        self.register_buffer('dbias', torch.tensor(0.))
        
    def update_stats(self, x):
        bs, nc, *_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0, 2, 3)
        s = x.sum(dims, keepdim=True)
        ss = (x*x).sum(dims, keepdim=True)
        c = self.count.new_tensor(x.numel()/nc)
        mom1 = 1 - (1-self.mom)/math.sqrt(bs-1)
        self.mom1 = self.dbias.new_tensor(mom1)
        self.sums.lerp_(s, self.mom1)
        self.sqrs.lerp_(ss, self.mom1)
        self.count.lerp_(c, self.mom1)
        self.dbias = self.dbias*(1-self.mom1) + self.mom1
        self.batch += bs
        self.step += 1
        
    def forward(self, x):
        if self.training: self.update_stats(x)
        sums = self.sums
        sqrs = self.sqrs
        c = self.count
        if self.step<100:
            sums = sums / self.dbias
            sqrs = sqrs / self.dbias
            c    = c    / self.dbias
        means = sums/c
        vars = (sqrs/c).sub_(means*means)
        if bool(self.batch < 20): vars.clamp_min_(0.01)
        x = (x-means).div_((vars.add_(self.eps)).sqrt())
        return x.mul_(self.gamma).add_(self.beta)

def conv_GReLU(n_inp, n_out, kernel_size=3, stride=2, btch_norm='running', **kwargs):
    layers = [nn.Conv2d(n_inp, n_out,
                       kernel_size,
                       padding=kernel_size//2,
                       stride=stride,
                       bias = not btch_norm),
              GReLU(**kwargs)]
    if btch_norm == 'regular': layers.append(RunningBatchNorm(n_out))
    if btch_norm == 'running': layers.append(RunningBatchNorm(n_out))
    
    return nn.Sequential(*layers)