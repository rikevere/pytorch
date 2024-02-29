import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification

def plotmodel(w1, w2, b):
    #o valor de X definido abaixo, serve para plotar a reta linear, baseado na expressão
    #são atribuídos a "x", 50 valores entre -2 e 4
    x = np.linspace(-2, 4, 50)
    #RELEMBRANDO EQUAÇÃO LINEAR
    # ax + by + c = 0
    # y = (-a*x - c)/b
    #w1*x1 + w2*x2 + b
    #w1*x1 + w2*x2 + w3*x3 b
    #então:
    # w1*x + w2*y + b = 0
    # y = (-w1*x - b)/w2
    y = (-w1*x -b)/w2
    print(y[19])
    print(x[19])

    #plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, edgecolors='k')
    #pega os valores de limite da reta
    xmin, xmax = plt.gca().get_xlim()
    ymin, ymax = plt.gca().get_ylim()


    #a função matplotlib é uma biblioteca de visualização, para podermos visualizar a nossa reta. plt(x, y)
    # Adicionar título e rótulos aos eixos
    plt.title("Gráfico de Exemplo Aula 3.03 - Classificação Linear")
    plt.axvline(0, -1, 1, color='k', linewidth=1)
    plt.axhline(0, -2, 4, color='k', linewidth=1)
    plt.grid(True)
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")

    # Criar linha do gráfico 
    plt.plot(x, y)

    #limita a área do Gráfico nos limites de X e Y
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    # Exibe o Gráfico
    # plt.show()

#função para classificar um ponto dados os pesos do modelo
def classify(ponto, w1, w2, b):
    ret = w1 * ponto[0] + w2 * ponto[1] + b
    if ret >= 0:
        return 1, 'yellow'
    else:
        return 0, 'blue'


# Gerar um conjunto de dados sintético
X, Y = make_classification(random_state=42, n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_clusters_per_class=1)

# X é um array de características e y é um array de rótulos de classe
print(X.shape)  # Saída: (100, 20)
print(Y.shape)  # Saída: (100,)

w1 = 5 #a da equação da reta "aula 2"
w2 = 1 #b da equação da reta
b = -3 #c da equação da reta

plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, edgecolors='k')

#chama a função para gerar a reta
plotmodel(w1, w2, b)

#define ponto para ser analisada sua posição em função da reta
#aqui estou referenciando um dos valores gerados para X
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
        print(acertos)
        acertos += 1

print("Acurácia: {0}".format(100*acertos/len(X)))

#mostra o gráfico
plt.show()


