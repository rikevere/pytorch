import torch 
from torch import nn, optim #bliblioteca de rede linear e implementação dos otimizadores
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader # Separa os dados em batches, embaralha os dados para treinamento e carrega dados em paralelo com threads

args = {
    'batch_size': 20,
    'num_workers': 4,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

print(f'Dispositivo de processamento: {args["device"]}')

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

    #Podemos então iterar no dataset para observar algumas amostras e seus rótulos
    for batch in train_loader:
        dado, rotulo = batch
        print(dado.size(), rotulo.size())
        #Para garantir que a imagem seja corretamente visualizada,  usei .numpy().squeeze() para converter
        #o tensor da imagem para um array NumPy e remover quaisquer dimensões unitárias.
        plt.imshow(dado[5][0].numpy().squeeze(), cmap='gray') 
        plt.title('Rotulo: ' + str(rotulo[0]))
        break

    plt.show()
#O código que executa a lógica principal é colocado dentro de uma função main(), que é chamada apenas se o 
#script for o ponto de entrada principal (if __name__ == '__main__':). Isso previne que o código seja 
#executado nos processos filhos no Windows, evitando o erro.
if __name__ == '__main__':
    main()