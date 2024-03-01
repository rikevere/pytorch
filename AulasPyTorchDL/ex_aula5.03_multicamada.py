import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
import torch  
from torch import nn #pacote do PyTorch de neural networks
from torchsummary import summary
from AulasPyTorchDL.exaula505classeModule import MinhaRede

# DEFININDO O LOCAL DE PROCESSAMENTO GPU OU CPU:
if torch.cuda.is_available():
    devide = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Dispositivo de processamento: {device}')

#DEFININDO UMA ARQUITETURA PyTorch
# Gerar um conjunto de dados sintético para amostragem
X1, Y1 = make_moons(n_samples=300, noise=0.2)

#FORWARD - Entrada de dados
#quando for array, precisa converter com o "frum_numpy". Se fosse uma lista, utilizaria o torch.Tensor()
#por padrão, as redes precisam se carregadas com o tipo Float32
print(f'X1 Shape: {X1.shape}')
print(f'X1 Data Type: {X1.dtype}')
tensor = torch.from_numpy(X1).float()
print(f'Tensor: {tensor}')

input_size = 2 #Tamanho da camada de entrada
hidden_size = 16 #Tamanho da entrada da camada escondida
output_size = 1 #Tamanho da saída

#nn.Sequential
#O módulo "nn.Sequential" é um container onde se pode colocar múltiplos módulos. Ao Realizar um "forward" (ir para frente) 
#em um objeto Sequential", ele aplicará sequencialmente os módulos nele contidos para gerar uma saída
net = nn.Sequential(nn.Linear(in_features=input_size, out_features=hidden_size), # hidden (camada escondida)
                      nn.ReLU(), # ativação não linear - ver aula 4.02
                      nn.Linear(in_features=hidden_size, out_features=output_size))
net = net.to(device)
print(f'Net: {net}')

#importando classe módulo from exaula505classeModule import MinhaRede, para fazer similar ao nn.Sequential acima.
net1 = MinhaRede(input_size, hidden_size, output_size) ## vai iniciar o método definido como __init__()
net1 = net1.to(device)
print(f'Net1: {net1}')

#fazer a predição de entrada para o net, escrito com o nn.Sequential
pred = net(tensor)
print(f'Tamanho pred - nn.Sequential: {pred.size()}')

#fazer a predição de entrada para o net, escrito com o nn.Sequential
pred1 = net1(tensor)
print(f'Tamanho pred1 - nn.Modelu: {pred1.size()}')


plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolors='k')

print(f'X1 Data Type: {X1.dtype}')


plt.show()

