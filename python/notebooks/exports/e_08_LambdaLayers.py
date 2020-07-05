# module automatically generated from 08_LambdaLayers.ipynb
# to change this code, please edit the appropriate notebook and re-export, rather than editing this script directly

from exports.e_07_Annealing import *

def normalize(x, m, s): return (x-m)/s

def normalize_data(train, valid):
    m,s = train.mean(),train.std()
    return normalize(train, m, s), normalize(valid, m, s)

def MNISTDataWrapper():
    x_train, y_train, x_valid, y_valid = loadMNIST()
    x_train, x_valid = normalize_data(x_train, x_valid)

    train_ds, valid_ds = Dataset(x_train, y_train), Dataset(x_valid, y_valid)

    n_sampl, n_inp = x_train.shape
    n_out = 10
    n_hid = 50

    batch_size = 512

    return DataWrapper(*make_dls(train_ds, valid_ds, batch_size), n_out)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x): return self.func(x)
        
def flatten(x): return x.view(x.shape[0], -1)
def mnist_square(x): return x.view(-1 , 1, 28, 28)

def CNNModel(data_w, lr=0.3):
    n_inp, n_out = data_w.train_ds.x.shape[1], data_w.n_out
    
    model = nn.Sequential(
        Lambda(mnist_square),
        nn.Conv2d( 1, 8, 5, padding=2,stride=2), nn.ReLU(), # 14
        nn.Conv2d( 8,16, 3, padding=2,stride=2), nn.ReLU(), # 7
        nn.Conv2d(16,32, 3, padding=1,stride=2), nn.ReLU(), # 4
        nn.Conv2d(32,32, 3, padding=1,stride=2), nn.ReLU(), # 2
        nn.AdaptiveAvgPool2d(1),
        Lambda(flatten),
        nn.Linear(32,n_out)
    )
    
    return model, optim.SGD(model.parameters(), lr=lr)