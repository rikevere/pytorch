import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification

def plotmodel(w1, w2, b):
    #o valor de X definido abaixo, serve para gerar dados para a reta linear, baseado na expressão
    #são atribuídos a "x", 50 valores entre -2 e 4
    x = np.linspace(-2, 4, 50)
    #RELEMBRANDO EQUAÇÃO LINEAR
    # ax + by + c = 0
    # y = (-a*x - c)/b
    #w1*x1 + w2*x2 + b
    #w1*x1 + w2*x2 + w3*x3 b
    #então, substituímos as variáveis para gerar uma reta baseada nas entradas e na equação linear:
    # w1*x + w2*y + b = 0
    # y = (-w1*x - b)/w2
    y = (-w1*x -b)/w2
    print(y[19])
    print(x[19])

    #pega os valores de limite da reta para facilitar a visualização
    xmin, xmax = plt.gca().get_xlim()
    ymin, ymax = plt.gca().get_ylim()
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    #a função matplotlib é uma biblioteca de visualização, para podermos visualizar a nossa reta. plt(x, y)
    # Adicionar título e rótulos aos eixos
    plt.title("Gráfico de Exemplo Aula 3.03 - Classificação Linear")
    plt.axvline(0, -1, 1, color='k', linewidth=1)
    plt.axhline(0, -2, 4, color='k', linewidth=1)
    plt.grid(True)
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")

    # Plota no gráfico os valores de X e Y conforme equação linear
    plt.plot(x, y)

#função para classificar um ponto dados os pesos do modelo
#para cada ponto gerado aleatoriamente no gráfico, identifica se seu resultado na equação é
#positivo ou negativo, retornando sua posição em relação a reta e sua cor (rótulos de classe)
def classify(ponto, w1, w2, b):
    ret = w1 * ponto[0] + w2 * ponto[1] + b
    if ret >= 0:
        return 1, 'yellow'
    else:
        return 0, 'blue'


# Gerar um conjunto de dados sintético para amostragem
X, Y = make_classification(random_state=199, n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_clusters_per_class=1)

# X é um array de características e y é um array de rótulos de classe
print(X.shape)  # Saída: (100, 20)
print(Y.shape)  # Saída: (100,) - 100 valores de classe 0 ou 1 para cada valor de X

w1 = -90 #a da equação linear - Rotaciona para a esquerda se negativo e direita se positivo
w2 = 50 #b da equação linear
b = 0 #c da equação linear

# A função a seguir é utilizada para criar um gráfico de dispersão, que é um tipo de 
#gráfico utilizado para exibir valores ao longo de duas dimensões
#Cada ponto no gráfico de dispersão representa um par de valores (X e Y = X0 e X1), 
#tornando-o útil para observar a relação entre duas variáveis numéricas.
#Neste exemplo, X[:, 0] são os valores para o eixo x, e X[:, 1], para o eixo y
#Cor dos pontos (c): neste exemplo recebe o valor do rótulos de classe para cada ponto de amostragem
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, edgecolors='k')

#chama a função para gerar a reta
plotmodel(w1, w2, b)

#define ponto para ser analisada sua posição em função da reta
#aqui estou referenciando um dos valores gerados para X as coordenadas X e Y
p = (X[19,0], X[19,1])

#função classify recebe a classe e a cor, de acordo com a classificação em função da reta
#conciderando um relógio, valores negativos ficam a esquerda (anti horário) e positivos a direita (horário)
classe, cor = classify(p, w1, w2, b)

#imprime o ponto de validação no gráfico
plt.plot(p[0], p[1], marker='^', markersize=5, color='g')
print(classe, cor)
if classe == Y[19]:
    print("Acertou")
else:
    print(f"errou, classe de Y = {Y[19]}")


#função para percorrer todos os valores de "X" na validação acima. 
#"range" cria números sequenciais de acordo com o tamanho "len" de X
acertos = 0
for k in range(len(X)):
    categ, _ = classify(X[k], w1, w2, b)
    if categ == Y[k]:
        acertos += 1

print("Acurácia: {0}".format(100*acertos/len(X)))

#mostra o gráfico
plt.show()


