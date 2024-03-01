import torch
from torch import nn
import torch.nn.functional as F


class MinhaRede(nn.Module):
    def __init__(self,entrada=7,camada_escondida1=14,camada_escondida2=49,saida=3):
        super().__init__()
        self.fc1 = nn.Linear(entrada,camada_escondida1)
        self.fc2 = nn.Linear(camada_escondida1, camada_escondida2)
        self.out = nn.Linear(camada_escondida2, saida)

    #Ainda dentro da classe modelo definimos a função forward que será responsável pela propagação da rede. 
    #A propagação é o que leva a entrada até a saída. Cada uma das conexões de uma rede como a da figura de exemplo 
    #é ligada através de pesos e a saída de cada camada da rede é feita usando uma função de ativação. 
    #Aqui vamos utilizar a função de ativação ReLU que retorna sempre valores positivos. A nossa saída da rede é sempre 
    #positiva e por este motivo é conveniente utilizar a ReLU.
            
    def forward(self, x):
        # Gerar uma saída a partir do X
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
    
