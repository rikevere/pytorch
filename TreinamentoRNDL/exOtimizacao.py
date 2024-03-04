import torch 
from torch import nn 
from torch import optim #possui implementação dos otimizadores
import torch.nn.functional as F
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler #processo de normalização dos dados para deichalos na mesma escala. Importante sempre normalizar
import numpy as np


# DEFININDO O LOCAL DE PROCESSAMENTO GPU OU CPU:
if torch.cuda.is_available():
    device = torch.device('cuda')  # Aqui deve ser 'device', não 'devide'
else:
    device = torch.device('cpu')
print(f'Dispositivo de processamento: {device}')

#Selecionando dados do dataset referente a Teor Alcoólico: Índice 0 e Intensidade da Cor: Índice 0
fetures = [0, 9]

#CAPTURANDO DADOS DO DATASET
wine = datasets.load_wine()
data = wine.data[:, fetures]
target = wine.target

scaler = StandardScaler()
data = scaler.fit_transform(data)

#plt.scatter(data[:, 0], data[:,1], c=target, cmap=plt.cm.brg)
#plt.xlabel(wine.feature_names[fetures[0]])
#plt.ylabel(wine.feature_names[fetures[1]])


input_size  = data.shape[1]
hidden_size = 32
out_size    = len(wine.target_names) #retorna o numero de classes

class indicadorDediabetes(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(indicadorDediabetes, self).__init__()
        # Define a estrutura da rede
        self.hidden = nn.Linear(input_size, hidden_size)  # Uma camada intermediária
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_size, out_size)       # Uma camada de saída
        self.softmax = nn.Softmax(dim=-1)
         
        #super(indicadorDediabetes, self).__init__()
        # Define a estrutura da rede
        #self.hidden = nn.Linear(input_size, hidden_size)  # Uma camada intermediária
        #self.out = nn.Linear(hidden_size, out_size)       # Uma camada de saída

    def forward(self, X):
        feature = self.relu(self.hidden(X))
        # Aplica a camada de saída e depois softmax na dimensão correta
        #output = self.softmax(self.out(feature))  
        output = self.out(feature)  # Corrigido aqui

        return output
    
def plot_boundary(X, y, model):
    # Define os limites do gráfico
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    
    spacing = min(x_max - x_min, y_max - y_min) / 100

    # Gera uma grade de pontos com uma distância .01 entre eles
    xx, yy = np.meshgrid(np.arange(x_min, x_max, spacing),
                         np.arange(y_min, y_max, spacing))
    
    data = (np.hstack((xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1))))
    #data = nn.hstack((xx.ravel().reshape(-1,1),
     #                 yy.ravel().reshape(-1,1)))
    
    db_prob = model(torch.Tensor(data).to(device))
    clf = np.argmax(db_prob.cpu().data.numpy(), axis =-1)

    Z = clf.reshape(xx.shape)

    # Plota o contorno e os exemplos de treinamento
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.brg)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=25, edgecolor='k', cmap=plt.cm.brg)
    plt.show()

net = indicadorDediabetes(input_size, hidden_size, out_size).to(device) #define se GPU ou CPU
net = net.to(device)

# Função de Perda

#criterion = nn.CrossEntropyLoss().to(device)

# otimizador
#Stochastic Gradient Descent SGD
#lr = Taxa de aprendizado - Definido pelo programador
#optimizer = optim.SGD(net.parameters(), lr=1e-3)

plot_boundary(data, target, net)












#VISUALIZANDO OS DADOS DO DATASET
#Dataset com 178 dados e 3 classes.
print(data.shape, target.shape)
print(wine.feature_names, wine.target_names)

class ClassificadorDeVinhos(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(ClassificadorDeVinhos, self).__init__()
        # Define a estrutura da rede
        self.hidden = nn.Linear(input_size, hidden_size)  # Uma camada intermediária
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_size, out_size)       # Uma camada de saída
        self.softmax = nn.Softmax()
         
        #super(ClassificadorDeVinhos, self).__init__()
        # Define a estrutura da rede
        #self.hidden = nn.Linear(input_size, hidden_size)  # Uma camada intermediária
        #self.out = nn.Linear(hidden_size, out_size)       # Uma camada de saída

    def forward(self, X):
        feature = self.relu(self.hidden(X))
        # Aplica a camada de saída e depois softmax na dimensão correta
        output = self.softmax(self.out(feature))  # Corrigido aqui
        #output = self.out(feature)  # Corrigido aqui

        return output

    #def relu(self, input):
        # Uma ativação não linear ReLU
    #    return F.relu(input)
    
input_size  = data.shape[1]
hidden_size = 32
out_size    = len(wine.target_names) 

net = ClassificadorDeVinhos(input_size, hidden_size, out_size).to(device) #define se GPU ou CPU
print(net)

#realizando a instancia da função de perda. Em uma classificação de 3 classes, uma função de entropia cruzada é recomendada "nn.CrossEntropyLoss()"
criterios = nn.CrossEntropyLoss().to(device)

#antes de aplicar a função de perda, é necessário fazer o cast dos dados para tensores e extrair as predições 'y' da rede.
Xtns = torch.from_numpy(data).float()
Ytns = torch.from_numpy(target).long()
#print(Xtns)

#cast na GPU
Xtns = Xtns.to(device)
Ytns = Ytns.to(device)

print(Xtns.dtype, Ytns.dtype)

#realizando a predição de 'y'
predicao = net(Xtns)
print(predicao.shape, Ytns.shape)
#irá entregar um valor escalar para indicar o erro (média) nos dados fornecidos (conjunto de dados).
loss = criterios(predicao, Ytns)
print(loss)


