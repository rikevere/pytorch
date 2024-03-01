import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # Embora não usado explicitamente, é necessário para projeção 3D
import torch  
from sklearn.datasets import make_classification
from torch import nn

# Gerar um conjunto de dados sintético para amostragem
X, Y = make_classification(random_state=46, n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_clusters_per_class=1)

def plotmodel(w1, w2, b):
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, edgecolors='k')
    #pega os valores de limite da reta para facilitar a visualização
    xmin, xmax = plt.gca().get_xlim()
    ymin, ymax = plt.gca().get_ylim()
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    
    x = np.linspace(-2, 4, 50)
    y = (-w1*x -b)/w2

    plt.title("Gráfico de Exemplo Aula 4.02 - Ativações Pytorch")
    plt.axvline(0, -1, 1, color='k', linewidth=1)
    plt.axhline(0, -2, 4, color='k', linewidth=1)
    plt.grid(True)
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")

    # Plota no gráfico os valores de X e Y conforme equação linear
    plt.plot(x, y)



w1 = 5 #a da equação linear - Rotaciona para a esquerda se negativo e direita se positivo
w2 = 1 #b da equação linear - Diminui ou aumenta o angulo de rotação (diferente de zero)
b = 1.2 #c da equação linear - desloca para cima ou para baixo

p = (-1, 1)
print(w1*p[0] + w2*p[1] + b)

#abaixo estamos instanciando a rede neural definindo as entradas e saídas
#perceptron = nn.Linear(in_features=2, out_features=1) - out_features = quantidade de nerônios
perceptron = nn.Linear(2,1)


#Fixa vao=lores para os W e b da função
perceptron.weight = nn.Parameter(torch.Tensor([[w1,w2]]))
perceptron.bias = nn.Parameter(torch.Tensor([b]))

print(perceptron.weight.data)
print(perceptron.bias.data)

#facilita o preenchimento de símbolos e cores para os pontos analisados
markers = ['^', 'v', '>', '<']
colors = ['r', 'g', 'b', 'gray']

plt.figure(figsize=(8,6))
#plota no gráfico os pontos aleatórios de amostragem
plotmodel(w1, w2, b)


for k, idx in enumerate([35, 41, 59, 73]):
    x = torch.Tensor(X[idx])
    print(f'torch.Tensor(X[idx]): {x}')
    #ret recebe o linear instanciado em perceptron de x
    ret = perceptron(x)
    print(f'ret = perceptron(x): {ret.data.numpy()[0]}')
    #act recebe o retorno ativado  da função sigmoide do que foi alimentado pelo perceptron
    #ativando a saída do perceptron
    #as ativações pode ser de diferentes tipos, conforme documentação Torch:
    #activation = nn.Sigmoid() #A identificação dos pontos é definida como 0 ou 1, dependendo do lado da reta
    activation = nn.ReLU() #A identificação dos pontos é definida com zero para quem for negativo mantém os valores positivos
    #activation = nn.Tanh() #A identificação dos pontos -e definida positivo ou negativo, dependendo do lado da reta
    act = activation(ret)
    print(f'act = sigmoide(ret): {act.data.numpy()[0]}')
    #compara o que o perceptron retornou, o que o limiar retornou e o que uma função linear retornaria
    act_limiar = 0 if ret.data < 0 else 1

    label = ' ret: {:5.2f}'.format(ret.data.numpy()[0]) + ' liminar: {:5.2f}'.format(act_limiar) + ' act: {:5.2f}'.format(act.data.numpy()[0])
    plt.plot(x[0], x[1], marker=markers[k], color=colors[k], markersize=10, label=label)

plt.legend()
#mostra o gráfico
plt.show()

