import torch
import numpy as np

#Tipos de tensores
#Você pode criar tensores do PyTorch de inúmeras formas! Vamos ver primeiro os tipos de tensores que estão ao nosso dispor. 
#Para isso, vamos converter listas comuns do Python em tensors do PyTorch.
#Note que a impressão de tensores dos tipos float32 e int64 não vêm acompanhadas do parâmetro de tipo dtype, visto que se tratam dos tipos padrão trabalhados pelo PyTorch.

lista = [ [1,2,3],
          [4,5,6] ]

tns = torch.Tensor(lista)
print(tns.dtype)
print(tns)

print('')
tns = torch.DoubleTensor(lista)
print(tns.dtype)
print(tns)

print('')
tns = torch.LongTensor(lista)
print(tns.dtype)
print(tns)

print('Outras formas de instanciar tensores')
#Outras formas de instanciar tensores
#A partir de arrays Numpy
print('Tensor array numpy')
arr = np.random.rand(3,4)
arr = arr.astype(int)
print(arr)
print(arr.dtype)

print('Tensor convertido de numpy para Pytorch')
tns = torch.from_numpy(arr)
print(tns)
print(tns.dtype)

#Tensores inicializados
#Essas funções recebem como parâmetro o tamanho de cada dimensão do tensor. Aqui vamos conhecer as seguintes funções:
#torch.ones() -> Cria um tensor preenchido com zeros.
#torch.zeros() -> Cria um tensor preenchido com uns.
#torch.randn() -> Cria um tensor preenchido com números aleatórios a partir de uma distribuição normal.
print('')
print('Tensores pre inicializados')
tns1 = torch.ones(2, 3)
tns0 = torch.zeros(3, 5)
tnsr = torch.randn(3, 3)
print('Tensores inicializado com 1')
print(tns1)

#Tensor para array numpy
print('')
print('Tensor convertido de Pytorch para numpy')
arr = tnsr.data.numpy()
print(arr)


#Indexação
#De posse dessa informação, a indexação é feita de forma similar a arrays Numpy, através da sintaxe de colchetes [].
print('')
print('Indexador de Tensores')
print('')
#tnsr[0,1]

print(tnsr[0:2, 2].data.numpy())
print(tnsr[0, 1].item())


#Operações com tensores
#A função .item() utilizada anteriormente extrai o número de um tensor que possui um único valor, permitindo realizar as operações numéricas do Python. 
#Caso o item não seja extraído, operações que envolvam tensores vão retornar novos tensores.


print('')
print('Operações com tensores')

tns10 = torch.randn(2,2)
tns20 = torch.ones(2,2)

tns30 = [ [1,2],
          [3,4] ]

tns40 = [ [2,2],
          [2,2] ]


tns30 = torch.Tensor(tns30)
tns40 = torch.Tensor(tns40)

print('')
print('Tensor randomico')
print(tns10)
print('')
print('Tensor com numeros 1')
print(tns20)
print('')
print('Tensor 30 - 2/2 com numeros de 1 a 4')
print(tns30)
print('')
print('Tensor 40 - 2/2 com numeros 2')
print(tns30)
print('')
print('Multiplicação tensor 30 pelo 40')
print(tns30*tns40)
print('')
#Função .size() e .view()
#Uma operações importantíssima na manipulação de tensores para Deep Learning é a reorganização das suas dimensões. 
#Dessa forma podemos, por exemplo, linearizar um tensor n-dimensional.

print(tns20.size())
print(tns20.view(tns20.size(0), -1))


#GPU Cast
#Para que o seu script dê suporte a infraestruturas com e sem GPU, é importante definir o dispositivo no início do seu código de acordo com a verificação apresentada a seguir. 
#Essa definição de dispositivo será utilizada toda vez que precisarmos subir valores na GPU, como os pesos da rede, os gradientes, etc.

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
  device = torch.device('cpu')
  
print(device)
tns20 = tns20.to(device)
print(tns20)
