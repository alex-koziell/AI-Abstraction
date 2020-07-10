# module automatically generated from 11_MoreStats.ipynb
# to change this code, please edit the appropriate notebook and re-export, rather than editing this script directly

def append_stats_hist(hook, model, inp, out):
    if not hasattr(hook, 'stats'): hook.stats = ([],[],[])
    means, stds, hists = hook.stats
    means.append(out.data.mean().cpu())
    stds .append(out.data.std().cpu())
    hists.append(out.data.cpu().histc(bins=40, min=0, max=10)) # a histogram telling us how many activations in each bin

def get_hist(h): return torch.stack(h.stats[2]).t().float().log1p()
# rows = bins, cols= batch no, value = number of activations

def get_max_min_hist(h, mn=True):
    h1 = torch.stack(h.stats[2]).t().float()
    return (h1[:2] if mn else h1[-2:]).sum(0)/h1.sum(0)