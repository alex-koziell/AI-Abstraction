# module automatically generated from 04_DataAPI.ipynb
# to change this code, please edit the appropriate notebook and re-export, rather than editing this script directly

class Dataset():
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]