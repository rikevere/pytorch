import torch 
from torch import nn, optim #bliblioteca de rede linear e implementação dos otimizadores
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader # Separa os dados em batches, embaralha os dados para treinamento e carrega dados em paralelo com threads
import numpy as np

args = {
    'batch_size': 20,
    'num_workers': 4,
    'num_classes': 10,
    'lr': 1e-4,
    'weight_decay': 5e-4,
    'num_epochs': 30,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

}

#print(f'Dispositivo de processamento: {args["device"]}')

def main():
    #'./' local para salvar o data set #carrega os dados de Treino "train=True"
    train_set = datasets.MNIST('./', train=True, transform=transforms.ToTensor(), download=True)
    #carrega os dados amostra Teste "train=false"
    test_set = datasets.MNIST('./', train=False, transform=transforms.ToTensor(), download=False)
    print('Amostras de Treino:', len(train_set), '\nAmostras de Teste:', len(test_set))
    train_loader = DataLoader(train_set, #recebe um dataset
                            batch_size=args['batch_size'], #recebe o tamanho do bach_size que definimos no início.
                            shuffle=True, #define se os dados serão ou não embaralhados
                            num_workers=args['num_workers']) #define quantos batch serão executados em paralelo "threads"

    test_loader = DataLoader(test_set, #recebe um dataset
                            batch_size=args['batch_size'], #recebe o tamanho do bach_size que definimos no início.
                            shuffle=True, #define se os dados serão ou não embaralhados
                            num_workers=args['num_workers']) #define quantos batch serão executados em paralelo "threads"
    
    input_size  = 28 * 28
    hidden_size = 128
    out_size    = 10 #quantidade de classes

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
            self.softmax    =   nn.Softmax()

        def forward(self, X):
            #redimencionar a amostra para somente uma dimenção
            X = X.view(X.size(0), -1)
            feature = self.features(X)
            # Aplica a camada de saída e depois softmax na dimensão correta
            #output = self.softmax(self.out(feature))  
            output = self.out(feature)  # Corrigido aqui

            return output

    #INICIALIZANDO A REDE
    net = MLP(input_size, hidden_size, out_size).to(args['device'])

    #DEFININDO LOSS E OTIMIZADOR
    criterion = nn.CrossEntropyLoss().to(args['device'])
    optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    conta=0
    
    for epoch in range(args['num_epochs']):
        conta += 1
        epoch_loss = [] 
        #Podemos então iterar no dataset para observar algumas amostras e seus rótulos
        print(f'Tamanho train_loader: {len(train_loader)}')
        print(f'Epoca: {conta}')
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

        print(f'loss: {loss}')
        epoch_loss = np.asarray(epoch_loss)
        print(f'Epoca {epoch}, Loss: {epoch_loss.mean()}, Desvio: {epoch_loss.std()}') 

    plt.show()
#O código que executa a lógica principal é colocado dentro de uma função main(), que é chamada apenas se o 
#script for o ponto de entrada principal (if __name__ == '__main__':). Isso previne que o código seja 
#executado nos processos filhos no Windows, evitando o erro.
if __name__ == '__main__':
    main()