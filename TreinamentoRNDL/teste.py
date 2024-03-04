import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression

def plot_boundary(X, y, model):
    # Define os limites do gráfico
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Gera uma grade de pontos com uma distância .01 entre eles
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Preveja a função em todos os pontos da grade
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plota o contorno e os exemplos de treinamento
    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title("Fronteira de Decisão")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")


# Gerando um conjunto de dados fictício
X, y = make_moons(n_samples=100, noise=0.2, random_state=42)

# Treinando um modelo de regressão logística
model = LogisticRegression()
model.fit(X, y)

# Plotando a fronteira de decisão
plot_boundary(X, y, model)
plt.show()
