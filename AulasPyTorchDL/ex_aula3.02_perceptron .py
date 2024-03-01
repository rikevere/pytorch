import torch
import numpy as np
import matplotlib.pyplot as plt

a = -1
b = 4
c = 0.4

# ax + by + c = 0
# y = (-a*x - c)/b
def plotline(a, b, c):
    x = np.linspace(-2, 4, 50)
    #print('valores de um array com 50 posições entre -2 e 4')
    #print(x)
    print('')
    y = (-a*x - c)/b

    #a função matplotlib é uma biblioteca de visualização, para podermos visualizar a nossa reta. plt(x, y)
    # Adicionar título e rótulos aos eixos
    plt.title("Gráfico de Exemplo Aula 3")
    plt.axvline(0, -1, 1, color='k', linewidth=1)
    plt.axhline(0, -2, 4, color='k', linewidth=1)
    plt.grid(True)
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")

    # Criar gráfico de linha
    plt.plot(x, y)
    # Exibe o Gráfico
    #plt.show()

p1 = (2, 0.4)
p2 = (1, 0.6)
p3 = (3, -0.4)
ret1 = a*p1[0] + b*p1[1] + c
ret2 = a*p2[0] + b*p2[1] + c
ret3 = a*p3[0] + b*p3[1] + c

#CONSIDERANDO OS RETORNOS
#Se f(x) for igual a 0 = Define que o ponto está na reta
#Se f(x) for maior do que 0 = Define que o ponto está acima da reta
#Se f(x) for menor do que 0 = Define que o ponto está abaixo da reta
print('Valor para P1: %.2f' % ret1)
print('Valor para P2: %.2f' % ret2)
print('Valor para P3: %.2f' % ret3)


plotline(a ,b, c)
plt.plot(p1[0], p1[1], color='b', marker='o')
plt.plot(p2[0], p2[1], color='r', marker='o')
plt.plot(p3[0], p3[1], color='g', marker='o')
plt.show()

