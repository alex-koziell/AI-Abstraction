# module automatically generated from 06_Callbacks.ipynb
# to change this code, please edit the appropriate notebook and re-export, rather than editing this script directly

class DataWrapper():
    def __init__(self, train_dl, valid_dl, n_out):
        self.train_dl, self.valid_dl, self.n_out = train_dl, valid_dl, n_out
    
    # @property denotes a 'get' property of the class
    @property
    def train_ds(self): return self.train_dl.dataset
    
    @property
    def valid_ds(self): return self.valid_dl.dataset

def SimpleModel(data_w, lr=0.3, n_hid=50):
    n_inp, n_out = data_w.train_ds.x.shape[1], data_w.n_out
    
    model = nn.Sequential(nn.Linear(n_inp, n_hid), 
                          nn.ReLU(),
                          nn.Linear(n_hid, n_out))
    
    return model, optim.SGD(model.parameters(), lr=lr)

class ModelWrapper():
    def __init__(self, model, opt, loss_f, data_w):
        self.model, self.opt, self.loss_f, self.data_w = model, opt, loss_f, data_w

STAGES = ['begin_fit',
          'begin_epoch',
          'begin_batch',
          'after_loss',
          'after_backward',
          'after_step',
          'begin_valid',
          'after_epoch',
          'after_fit']

class Callback():
    _order = 0
    def __getattr__(self, attr): return getattr(self.job, attr)
    @property
    def name(self):
        return self.__class__.__name__

class DLJob():
    def __init__(self, callbacks=[]):
        self.cbs, self.stop = callbacks, False
    
    @property
    def opt(self): return self.mw.opt
    @property
    def model(self): return self.mw.model
    @property
    def loss_f(self): return self.mw.loss_f
    @property
    def data_w(self): return self.mw.data_w
    
    
    def one_batch(self, xb, yb):
        self.xb, self.yb = xb, yb
        if not self('begin_batch'): return
        self.pred = self.model(self.xb)
        if not self('after_pred'): return
        self.loss = self.loss_f(self.pred, self.yb)
        if not self('after_loss') or not self.training: return
        self.loss.backward()
        if not self('after_backward'): return
        self.opt.step()
        if not self('after_step'): return
        self.opt.zero_grad()
    
    def all_batch(self, dl):
        self.iters = len(dl)
        for xb, yb in dl:
            if self.stop: break
            self.one_batch(xb, yb)
            self('after_batch')
        self.stop = False
        
    def fit(self, epochs, model_wrapper):
        self.epochs, self.mw = epochs, model_wrapper
        
        try:
            for cb in self.cbs: cb.job = self
            if not self('begin_fit'): return
            for epoch in range(epochs):
                self.epoch = epoch
                if self('begin_epoch'): 
                    self.training = True
                    self.model.train()
                    self.all_batch(self.data_w.train_dl)
                    
                with torch.no_grad():
                    if self('begin_valid'):
                        self.training = False
                        self.model.eval()
                        self.all_batch(self.data_w.valid_dl)
                if not self('after_epoch'): break
                    
        finally:
            self('after_fit')
            self.mw = None
            
            
    def __call__(self, stage):
        for cb in sorted(self.cbs, key=lambda x: x._order):
            f = getattr(cb, stage, None)
            if f and f(): return False
        return True

class AvgStats():
    def __init__(self, metrics, training):
        self.metrics, self.training = metrics, training
    
    def reset(self):
        self.tot_loss, self.count = 0., 0
        self.tot_mets = [0.] * len(self.metrics)
    
    @property
    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets
    @property
    def avg_stats(self): return [stat/self.count for stat in self.all_stats]
    
    def __repr__(self):
        if not self.count: return ""
        return f'{"train" if self.training else "valid"}: {self.avg_stats}'
    
    def accumulate(self, job):
        batch_size = job.xb.shape[0]
        self.tot_loss += job.loss * batch_size
        self.count += batch_size
        for i, metric in enumerate(self.metrics):
            self.tot_mets[i] += metric(job.pred, job.yb) * batch_size
            
class AvgStatsCB(Callback):
    def __init__(self, metrics=[]):
        self.train_stats, self.valid_stats = AvgStats(metrics, True), AvgStats(metrics, False)
    
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        
    def after_loss(self):
        stats = self.train_stats if self.training else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.job)
    
    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)

def acc_f(pred, lab): return (torch.argmax(pred, dim=1) == lab).float().mean()