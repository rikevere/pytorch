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

#Proceso a seguir realiza a padronização dos dados.
#sem a padronização, o processo de otimização tem maior dificuldade em encontrar um resultado adequado. 
#Isso ocorre porque a diferença na escala entre as variáveis dificulta a sua combinação pelo modelo. 
#Processos como a padronização e a normalização levam todos os dados para uma escala similar, enquanto mantêm as suas distribuições originais.
#Cria uma instância do StandardScaler. Este objeto é capaz de normalizar os dados de entrada subtraindo a média e dividindo pelo desvio padrão, 
#resultando em dados com média 0 e variância 1. Isso é conhecido como padronização ou z-score normalization.
scaler = StandardScaler()
data = scaler.fit_transform(data)   

#plt.scatter(data[:, 0], data[:,1], c=target, cmap=plt.cm.brg)
#plt.xlabel(wine.feature_names[fetures[0]])
#plt.ylabel(wine.feature_names[fetures[1]])


input_size  = data.shape[1]
hidden_size = 32
out_size    = len(wine.target_names) #retorna o numero de classes


# CLASSE QUE DEFINE A ESTRUTURA DA REDE

class classificavinho(nn.Module):
    #função que é inicializada na chamada da classe
    def __init__(self, input_size, hidden_size, out_size): #informados parâmetros da quantidade de neurônios da rede
        super(classificavinho, self).__init__()
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
    #plt.figure() # cria um gráfico novo
    plt.show() #exibe o gráfico

#MONTA A REDE A SER UTILIZADA
net = classificavinho(input_size, hidden_size, out_size).to(device) #Instancia a classe (rede) na variável NET e define se GPU ou CPU
net = net.to(device)

# INICIALIZA A FUNÇÃO DE PERDA - LOSS
criterion = nn.CrossEntropyLoss().to(device)

# INICIALIZA O OTIMIZADOR DE DADOS E A TAXA DE PERDA - CÁLCULO DA MÉDIA DE UM AGRUPAMENTO DE DADOS
#Stochastic Gradient Descent SGD
#lr = Taxa de aprendizado - Definido pelo programador
#mais abaixo, quando executado o optimizer.step(), o sistema saberá que devem ser ajustados os parâmetros da rede "net" instanciada
#HIPERPARÂMETO: 1e-3 = 0.001 = 1 x 10-3. Este valor se refere ao passo de decida do gradiente.
#Existem vários tipos de orimizadores. SGD, Momentum, NAG, ADAGRAD, ADADELTA, RMSPROP

#Descida do Gradiente
optimizer = optim.SGD(net.parameters(), lr=0.001)
#Descida do Gradiente + Momentum (Avalia a taxa da descida e se pode ser explorado mais na subida para novas descidas)
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #valor 0.9 é conciderado um bom índice
#Descida do Gradiente + Momentum + Nesterov
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True) #melhora o cálculo do momentum
#Descida do Gradiente + Momentum + Decaimento de Pesos
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4) #adequado para modelos mais simples e uso do overfeet
#optimizer = optim.RMSprop(net.parameters(), lr=0.001, weight_decay=5e-4, momentum=0.9)
#optimizer = optim.Adagrad(net.parameters(), lr=0.001, weight_decay=5e-4)
#optimizer = optim.Adadelta(net.parameters(), lr=0.001, weight_decay=5e-4)
#optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4) #Normalmente a melhor escolha de otimizador

#REALIZA O CAST DOS DADOS
#Os dados carregados do sklearn são dados retornados como arrays ((178, 2) (178,)), por isso precisamos convertê-los
#em tensores e carregá-los na GPU (se tiver e quiser) antes de alimentar o modelo neural
X = torch.FloatTensor(data).to(device) #converte o tipo do tensor e carrega da GPU se disponível (Ver IF do DEVICE)
#em Y sempre vai os rótulos (tipos de dados). Em X sempre vai os dados disponíveis para os rótulos 
#(DADOS: 178 em duas dimenções (178, 2) Rótulos/Tipos (targets): 178 em uma dimensão (178,))
Y = torch.LongTensor(target).to(device) 

for i in range(300): #Laço de repetição para que o FORWARD possa ser repetido por 100 ciclos
    #FORWARD - Passos que precisam ser dados pela rede para que se possa gaver treinamento:
    #1 - Alimentar os dados para a rede:
    predicao = net(X)
    #2 - Calcular a função de custo - O quão longe os dados estão de cada rótulo (referência na tangente)
    loss = criterion(predicao, Y)

    #BACKPROPAGATION - Retorna o valor médio do erro para que o ciclo se repita e os pesos w1, w2 e b sejam ajustados em cada ciclo de repetição
    #1 - Calcular o Gradiente - Cálculo da derivada da perda, para verifica se está mais próximo ou mais distante do ponto ideal
    loss.backward()
    #2 - A partir do gradiente calculado, atualiza os pesos - w1, w2 e b
    optimizer.step()

    if i % 50 == 0:
        #plot_boundary(data, target, net)
        print(f'step: {i} erro: {loss}')

plot_boundary(data, target, net)