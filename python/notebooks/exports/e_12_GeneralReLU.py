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

def append_stats_hist_GReLU(hook, model, inp, out):
    if not hasattr(hook, 'stats'): hook.stats = ([],[],[])
    means, stds, hists = hook.stats
    means.append(out.data.mean().cpu())
    stds .append(out.data.std().cpu())
    hists.append(out.data.cpu().histc(bins=40, min=-3, max=7)) # a histogram telling us how many activations in each bin