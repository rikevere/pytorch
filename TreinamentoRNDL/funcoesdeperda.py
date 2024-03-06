import torch
from torch import nn 
import torch.nn.functional as F
from sklearn import datasets

# DEFININDO O LOCAL DE PROCESSAMENTO GPU OU CPU:
if torch.cuda.is_available():
    device = torch.device('cuda')  # Aqui deve ser 'device', não 'devide'
else:
    device = torch.device('cpu')
print(f'Dispositivo de processamento: {device}')

#CAPTURANDO DADOS DO DATASET
wine = datasets.load_wine()
data = wine.data
target = wine.target

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


