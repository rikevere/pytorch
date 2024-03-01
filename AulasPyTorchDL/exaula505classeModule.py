import torch
from torch import nn
import torch.nn.functional as F

class MinhaRede(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MinhaRede, self).__init__()
    
        # Definir a arquitetura
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu   = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size) 


    def forward(self, X):
       
       # Gerar uma saída a partir do X
       hidden = self.relu(self.hidden(X))
       output = self.output(hidden)

       return output 