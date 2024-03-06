import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn 

# DEFININDO O LOCAL DE PROCESSAMENTO GPU OU CPU:
if torch.cuda.is_available():
    devide = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Dispositivo de processamento: {device}')

#Primeiro, precisamos carregar e preparar os dados para o treinamento. Isso inclui ler o arquivo CSV, 
#normalizar os dados (se necessário) e dividir os dados em conjuntos de treinamento e teste.

# Carregar dados
df = pd.read_csv('sementes.csv')

# Separar características e rótulos
X = df.drop('Espécie', axis=1).values
y = df['Espécie'].values

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Converter para tensores PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Criar DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

#Aqui, definimos a arquitetura do modelo de rede neural usando nn.Module.
class ClassificadorSementes(nn.Module):
    def __init__(self):
        super(ClassificadorSementes, self).__init__()
        self.layer1 = nn.Linear(X.shape[1], 64)  # Camada de entrada
        self.layer2 = nn.Linear(64, 32)  # Camada oculta
        self.layer3 = nn.Linear(32, 3)  # Camada de saída, 3 classes

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)  # Não aplicamos softmax aqui pois será incluído na loss
        return x

# Inicializamos a classe
modelo_classificacao = ClassificadorSementes()

# Definimos o otimizador e a função de perda para o treinamento.
otimizador = torch.optim.Adam(modelo_classificacao.parameters(), lr=0.01)
# Verificamos o erro para avaliar se está em direção ao desejado
funcao_objetivo = nn.CrossEntropyLoss()

#Executamos o loop de treinamento, ajustando os pesos do modelo com base nos dados de treinamento.

epocas = 100

for epoca in range(epocas):
    for X_batch, y_batch in train_loader:
        # Zerar gradientes
        otimizador.zero_grad()
        
        # Forward pass
        y_pred = modelo_classificacao(X_batch)
        
        # Calcular a perda
        loss = funcao_objetivo(y_pred, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Atualizar os pesos
        otimizador.step()
    
    if (epoca+1) % 10 == 0:
        print(f'Epocas: {epoca+1}, Loss: {loss.item()}')


#Após o treinamento, avaliamos o desempenho do modelo nos dados de teste.
with torch.no_grad():
    y_pred = modelo_classificacao(X_test_tensor)
    _, predicao = torch.max(y_pred, 1)
    acuracia = (predicao == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Acurácia: {acuracia * 100:.2f}%')


