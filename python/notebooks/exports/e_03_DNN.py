# module automatically generated from 03_DNN.ipynb
# to change this code, please edit the appropriate notebook and re-export, rather than editing this script directly

from torch import nn

class TorchDNN(nn.Module):
    def __init__(self, n_inputs, n_neurons, n_outputs):
        super().__init__()
        self.layers = [nn.Linear(n_inputs, n_neurons, n_outputs),
                       nn.ReLU(),
                       nn.Linear(n_neurons, n_outputs)]
        self.lossLayer = nn.MSELoss()
        
    def __call__(self, x, labels):
        for layer in self.layers:
            x = layer(x)
        self.loss = self.lossLayer(x.squeeze(), labels)
        return self.loss
    
    def backward(self):
        self.loss.backward()