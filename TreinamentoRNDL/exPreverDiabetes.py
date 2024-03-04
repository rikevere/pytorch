import torch
from torch import nn 
import torch.nn.functional as F
from sklearn import datasets

diabetes = datasets.load_diabetes()
dados = diabetes.data
rotulos = diabetes.target


# DEFININDO O LOCAL DE PROCESSAMENTO GPU OU CPU:
if torch.cuda.is_available():
    device = torch.device('cuda')  # Aqui deve ser 'device', não 'devide'
else:
    device = torch.device('cpu')
print(f'Dispositivo de processamento: {device}')


print(dados.shape, rotulos.shape)

print(dados[0], rotulos[0])

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

    #def relu(self, input):
        # Uma ativação não linear ReLU
    #    return F.relu(input)
    
print(f'Dados input_size: {dados.shape}')
print(f'Dados input_size: {rotulos.shape}')
print(f'Dados input_size: {dados.shape[1]}')

input_size  = dados.shape[1]
hidden_size = 32
out_size    = 1 #Somente uma saída pois só queremos um resultado no Final. A progressão da Diabetes

net = indicadorDediabetes(input_size, hidden_size, out_size).to(device) #define se GPU ou CPU
print(f'NET: {net}')

#critério para a função de perda é dizer o quanto longe estamos distante do resultado esperado
#aqui realiza a instância da função de perda na variável critério. Existem diferentes tipos de função de perda
criterio = nn.MSELoss().to(device)
#criterio = nn.L1Loss().to(device)

#antes de aplicar a função de perda, é necessário fazer o cast dos dados para tensores e extrair as predições 'y' da rede.
Xtns = torch.from_numpy(dados).float()
Ytns = torch.from_numpy(rotulos).long()


#cast na GPU
Xtns = Xtns.to(device)
Ytns = Ytns.to(device)

print(f'Shape dos tensores: {Xtns.shape}, {Ytns.shape}')

pred = net(Xtns)
print(f'Shape do Tensores de Rótulos: {Ytns.shape}')
print(f'Shape dos Tensores depois "Forward" rede sem squeeze(): {pred.shape}')
print(f'Shape dos Tensores depois "Forward" rede com squeeze(): {pred.squeeze().shape}')

#Como o shape dos Tensores depois "Forward" da rede retornou mais de uma dimensão e quando você está avaliando um modelo com uma única amostra, 
#a saída pode ter uma dimensão extra, como neste caso. Usar squeeze() pode ajudar a remover essa dimensão extra para que a saída possa ser diretamente 
#comparada com o rótulo ou usada em cálculos subsequentes Ytns.shape.
loss = criterio(pred.squeeze(), Ytns)
print(f'Depois da função de Perda: {loss}')

