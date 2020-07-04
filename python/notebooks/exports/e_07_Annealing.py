# module automatically generated from 07_Annealing.ipynb
# to change this code, please edit the appropriate notebook and re-export, rather than editing this script directly

class Recorder(Callback):
    def begin_fit(self): self.lrs, self.losses = [], []
    
    def after_batch(self):
        if not self.training: return
        self.lrs.append(self.opt.param_groups[-1]['lr'])
        self.losses.append(self.loss.detach().cpu())
        
    def plot_lr(self): plt.plot(self.lrs)
    def plot_loss(self): plt.plot(self.losses)
        
class ParamScheduler(Callback):
    _order = 1
    def __init__(self, param, sched_func):
        self.param, self.sched_f = param, sched_func
    
    def set_param(self):
        for pg in self.opt.param_groups:
            pg[self.param] = self.sched_f(self.epoch/self.epochs)
    
    def begin_batch(self):
        if self.training: self.set_param()

def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner

@annealer
def sched_linear(start, end, pos): return start + pos*(end-start)

@annealer
def sched_cosine(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start)/2
@annealer
def sched_static(start, end, pos): return start
@annealer
def sched_exp(start, end, pos): return start * (end/start)**pos

def combine_schedules(pcts, scheds):
    assert sum(pcts) == 1
    pcts = torch.tensor([0] + pcts)
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        if idx == pcts.shape[0]-1: idx -= 1
        actual_pos = (pos - pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner