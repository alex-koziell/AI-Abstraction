# module automatically generated from 05_Losses_Optimizers_TrainEval.ipynb
# to change this code, please edit the appropriate notebook and re-export, rather than editing this script directly

from torch.utils.data import DataLoader

def make_dls(train_ds, valid_ds, batch_size, **kwargs):
    return (DataLoader(train_ds, batch_size, shuffle=True, **kwargs),
            DataLoader(valid_ds, batch_size*2, **kwargs))