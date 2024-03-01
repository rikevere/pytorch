import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # Embora não usado explicitamente, é necessário para projeção 3D
import torch  
from sklearn.datasets import make_classification
from torch import nn

#o comando acima define uma inicialização manual para por observar sempre o mesmo valor
#e não possuir respostas aleatórias
torch.manual_seed(42)


#criação de um perceptron de 3 entradas (x1, x2, x3) e uma única saída (y)
#perceptron = nn.Linear(in_features=3, out_features=1)
perceptron = nn.Linear(3,1)
print(perceptron)

#resultado do print: Linear(in_features=3, out_features=1, bias=True)
#os pesos W e o viés b são inicialisados aleatoriamente pelo pytorch. 
#Podemos consultar esta informação da seguinte forma:
for nome, tensor in perceptron.named_parameters():
    print(nome, tensor.data)
    #Outra opção: print(perceptron.weight.data, perceptron.bias.data)
    #retorno: weight tensor([[-0.1047, -0.3693,  0.1101]])
    #bias (viés) tensor([0.4505])

def plot3d(perceptron):
    #Este perceptron possui valores que definem um hiperplano no espaço de 3 dimensões
    #conforme definido na aula anterior, a equação seria
    # w1 * x1 + w2 * x2 + w3 * x3 = b = 0
    #observando o resultado dos valores do perceptron para w (weight), capturamos os valores da seguinte forma
    #como o perceptron é de 3 dimensões para uma, buscaremos com o numpy somente a primeira camada (numpy()[0])   
    w1, w2, w3 = perceptron.weight.data.numpy()[0]
    b = perceptron.bias.data.numpy()
    #procedimento a seguir limita o gráfico
    X1 = np.linspace(-1,1,10)
    X2 = np.linspace(-1,1,10)

    X1, X2 = np.meshgrid(X1,X2)

    X3 = (b - w1*X1 - w2*X2) / w3

    fig = plt.figure(figsize=(10,8))
    # O código a seguir realiza a criação explícita de um eixo 3D
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=180)
    ax.plot_surface(X1, X2, X3, cmap='plasma')

#X vai receber um tensor com valores fixos que serão fornecidos par a o perceptron
# que está instanciado com a camada linear "nn.linear"
X = torch.Tensor([0,1,2])

#y vai receber a saída do calculo linear do tensor X "linear(X)"
y = perceptron(X)
print(y)

#executa a função com os valores iniciais do perceptron
plot3d(perceptron)
#platando o ponto específico definido para X
plt.plot([X[0]], [X[1]], [X[2]], color='r', marker='^', markersize=20)
#mostra o gráfico
plt.show()