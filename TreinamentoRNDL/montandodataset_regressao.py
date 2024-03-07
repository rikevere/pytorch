import torch 
from torch import nn, optim #bliblioteca de rede linear e implementação dos otimizadores
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader # Separa os dados em batches, embaralha os dados para treinamento e carrega dados em paralelo com threads
import numpy as np
import pandas as pd

# Este exemplo vai demonstrar como pegar dados de um dataset (arquivo CSV com qualquer tipo de amostragem)

args = {
    'batch_size': 20,
    'num_workers': 4,
    'num_classes': 10,
    'lr': 1e-4,
    'weight_decay': 5e-4,
    'num_epochs': 30,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}


"""
#Processo para importar o arquivo macro e separar em dados de amostra e em dados de teste
#Neste exemplo está importando os dados do CSV hour.csv
# Substitua 'hour.csv' pelo caminho correto do arquivo se necessário
df = pd.read_csv('hour.csv')
print(len(df)) #Exibe o tamanho de amostras
print(df.head())  # Exibe as primeiras linhas do dataframe para verificação

torch.manual_seed(1) #fixa a montagem do tensor para sempre exibir o mesmo padrão
indices = torch.randperm(len(df)).tolist() #gera o embaralhamento dos dados (torch.randperm). cria uma permutação aleatória de inteiros no intervalo especificado
#no código acima, por exemplo (0, 1, 2, 3) pode virar (2, 0, 3, 1)
train_size = int(0.8*len(df)) #define em 80% o tamanho do treino dos dados disponíveis no arquivo
df_train    = df.iloc[indices[:train_size]] #define que os dados para df_train serão do início até o indice (0% a 80%)
df_test     = df.iloc[indices[train_size:]] #define que os dados para df_test serão do indice até o fim (80% a 100%)

print(len(df_train), len(df_test)) #mostra o tamanho de cada dado de amostragem

#separa os dados para teste e treino em dois novos arquivos, já com os dados embaralhados
df_train.to_csv('bike_train.csv', index=False) #index=False eminina os ID de linha pois não são relevantes
df_test.to_csv('bike_test.csv', index=False)
"""

#01 - CRIAR A CLASSE DATASET - PARA DADOS QUE VIERAM DE UM CSV
# A criação de uma classe Dataset personalizada a partir de amostras de um CSV no PyTorch permite a manipulação estruturada dos dados, 
#facilitando a leitura, pré-processamento e transformação dos dados de forma eficiente e integrada ao ecossistema PyTorch, 
#garantindo que os dados estejam prontos e no formato adequado para treinamento de modelos de aprendizado de máquina.
class bikes(Dataset):
    def __init__(self, caminho_csv): #Aqui será definido onde encontrar todas as amostras
        self.dados = pd.read_csv(caminho_csv).to_numpy()

    def __getitem__(self, idx): #idx se refere a uma amostra, onde o torch aplica as devidas transformações e retorna uma tupla (dados, rotulo)
        amostra = self.dados[idx][2:14]  # "amostra" vai receber dados da coluna 2 a 14 [2,14] do índice informado, pois são os dados desejaveis da amostra. Exclui indice a resposta (rotulo)
        rotulo = self.dados[idx][-1:]

        #converter para tensor
        amostra = torch.from_numpy(amostra.astype(np.float32))
        rotulo  = torch.from_numpy(rotulo.astype(np.float32))

        return amostra, rotulo
    
    #retornar o tamanho da amostra
    def __len__(self):
        return len(self.dados)
    
train_set   = bikes('bike_train.csv')
test_set    = bikes('bike_test.csv')

dado, rotulo = train_set[2] #acessa uma amostra amostra (índice n) do conjunto de dados de treinamento (train_set) criado a partir da classe Dataset personalizada
#print(rotulo)
#print(dado)

#02 - CRIAR UM DATALOADER PARA PROCESSAR O DATASET CRIADO
#O DataLoader no PyTorch automatiza o processo de carregamento dos dados, permitindo o carregamento eficiente de grandes conjuntos 
#de dados em lotes (batches), e oferece funcionalidades como embaralhamento (shuffle) dos dados e carregamento paralelo usando múltiplas threads, 
#facilitando e otimizando o treinamento de modelos de aprendizado de máquina em larga escala.
def main():
    train_loader = DataLoader(train_set, #recebe um dataset
                            batch_size=args['batch_size'], #recebe o tamanho do bach_size que definimos no início.
                            shuffle=True, #define se os dados serão ou não embaralhados
                            num_workers=args['num_workers']) #define quantos batch serão executados em paralelo "threads"

    test_loader = DataLoader(test_set, #recebe um dataset
                            batch_size=args['batch_size'], #recebe o tamanho do bach_size que definimos no início.
                            shuffle=True, #define se os dados serão ou não embaralhados
                            num_workers=args['num_workers']) #define quantos batch serão executados em paralelo "threads"
    
    input_size  = len(train_set[0][0])
    hidden_size = 128
    out_size    = 1 #Número de variáveis a serem preditas. Neste caso das bicicletas, somente uma. Pois são quantas bikes serão aligadas

    class MLP(nn.Module):
        #função que é inicializada na chamada da classe
        def __init__(self, input_size, hidden_size, out_size): #informados parâmetros da quantidade de neurônios da rede
            super(MLP, self).__init__()
            # Define a estrutura da rede
            self.features   =   nn.Sequential(
                                nn.Linear(input_size, hidden_size),       # Uma camada de saída
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),       # Uma camada de saída
                                nn.ReLU()
            )
            self.out        =   nn.Linear(hidden_size, out_size)


        def forward(self, X):
            feature = self.features(X)
            output = self.out(feature)

            return output

    #INICIALIZANDO A REDE
    net = MLP(input_size, hidden_size, out_size).to(args['device'])

    #DEFININDO LOSS E OTIMIZADOR
    criterion = nn.L1Loss().to(args['device'])
    optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    #TRANSFORMANDO O TREINAMENTO EM UMA FUNÇÃO
    def treino(train_loader, net, epoch):
        net.train()  
        epoch_loss = [] 
        #Podemos então iterar no dataset para observar algumas amostras e seus rótulos
        for batch in train_loader:
            dado, rotulo = batch
            
            #CAST na GPU
            dado = dado.to(args['device'])
            rotulo = rotulo.to(args['device'])

            #FORWARD
            pred = net(dado)
            loss = criterion(pred, rotulo)
            epoch_loss.append(loss.cpu().data)  

            #BACKWARD
            loss.backward()
            optimizer.step()

        print(f'loss de treino: {loss}')
        epoch_loss = np.asarray(epoch_loss)
        print(f'Epoca de treino {epoch}, Loss de treino: {epoch_loss.mean()}, Desvio de treino: {epoch_loss.std()}') 

    #TRANSFORMANDO O TREINAMENTO EM UMA FUNÇÃO
    def teste(test_loader, net, epoch):
        net.eval()
        with torch.no_grad():
            epoch_loss = [] 
            #Podemos então iterar no dataset para observar algumas amostras e seus rótulos
            for batch in test_loader:
                dado, rotulo = batch
                
                #CAST na GPU
                dado = dado.to(args['device'])
                rotulo = rotulo.to(args['device'])

                #FORWARD
                pred = net(dado)
                loss = criterion(pred, rotulo)
                epoch_loss.append(loss.cpu().data)  

            print(f'loss de teste: {loss}')
            epoch_loss = np.asarray(epoch_loss)
            print(f'Epoca de teste {epoch}, Loss de teste: {epoch_loss.mean()}, Desvio de teste: {epoch_loss.std()}') 

    for epoch in range(args['num_epochs']):
        treino(train_loader, net, epoch)
        teste(train_loader, net, epoch)

#O código que executa a lógica principal é colocado dentro de uma função main(), que é chamada apenas se o 
#script for o ponto de entrada principal (if __name__ == '__main__':). Isso previne que o código seja 
#executado nos processos filhos no Windows, evitando o erro.
if __name__ == '__main__':
    main()


