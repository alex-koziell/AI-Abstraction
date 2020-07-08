# module automatically generated from 09_Hooks.ipynb
# to change this code, please edit the appropriate notebook and re-export, rather than editing this script directly

from exports.e_08_LambdaLayers import *

def conv_relu(n_inp, n_out, kernel_size=3, stride=2):
    return nn.Sequential(
            nn.Conv2d(n_inp, n_out,
                      kernel_size,
                      padding=kernel_size//2,
                      stride=stride),
            nn.ReLU())

    
def cnn_layers(data_w, n_kernels):
    n_kernels = [1] + n_kernels
    
    return [
        conv_relu(n_kernels[i], n_kernels[i+1], 5 if i==0 else 3)
        for i in range(len(n_kernels)-1)
    ] + [nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(n_kernels[-1], data_w.n_out)]

class Hook():
    def __init__(self, m, f): self.hook = m.register_forward_hook(partial(f, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()
        
def append_stats(hook, model, inp, out):
    if not hasattr(hook, 'stats'): hook.stats = ([],[])
    means, stds = hook.stats
    means.append(out.data.mean())
    stds .append(out.data.std())

class ListContainer():
    def __init__(self, items): self.items = list(items)
    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)): return self.items[idx]
        if isinstance(idx[0], bool):
            assert len(idx) == len(self)
            return [o for m,o in zip(idx, self.items) if m]
        return [self.items[i] for i in idx]
    def __len__(self): return len(self.items)
    def __iter__(self): return iter(self.items)
    def __setitem__(self, i, o): self.items[i] = o
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self):
        rep = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self) > 10: rep = rep[:-1] + '...]'
        return rep

class Hooks(ListContainer):
    def __init__(self, ms, f): super().__init__([Hook(m, f) for m in ms])
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.remove()
    def __del__(self): self.remove()
        
    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(self, i)
        
    def remove(self):
        for h in self: h.remove()