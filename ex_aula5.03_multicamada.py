import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
import torch  
from torch import nn #pacote do PyTorch de neural networks
from torchsummary import summary

#DEFININDO UMA ARQUITETURA PyTorch

# Gerar um conjunto de dados sintético para amostragem
X1, Y1 = make_moons(n_samples=300, noise=0.2)

#FORWARD - Entrada de dados
#print(X1.datatype)

input_size = 2 #Tamanho da camada de entrada
hidden_size = 16 #Tamanho da entrada da camada escondida
output_size = 1 #Tamanho da saída

#nn.Sequential
#O módulo "nn.Sequential" é um container onde se pode colocar múltiplos módulos. Ao Realizar um "forward" (ir para frente) 
#em um objeto Sequential", ele aplicará sequencialmente os módulos nele contidos para gerar uma saída
net = nn.Sequential(nn.Linear(in_features=input_size, out_features=hidden_size), # hidden (camada escondida)
                      nn.ReLU(), # ativação não linear - ver aula 4.02
                      nn.Linear(in_features=hidden_size, out_features=output_size))

#quando for array, precisa converter com o "frum_numpy". Se fosse uma lista, utilizaria o torch.Tensor()
#por padrão, as redes precisam se carregadas com o tipo Float32
tensor = torch.from_numpy(X1).float()

#fazer a predição de entrada
pred = net(tensor)
print(pred.size())

plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolors='k')


plt.show()

