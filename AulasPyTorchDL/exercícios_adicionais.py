import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from classeModeloexSementes import MinhaRede


dados = pd.read_csv('sementes.csv')
dados.head()

X = dados.drop(['Espécie'],axis=1).values
y = dados['Espécie'].values

X_treino,X_teste,y_treino,y_teste = train_test_split(X,y,test_size=0.2)

X_treino = torch.FloatTensor(X_treino)
X_teste = torch.FloatTensor(X_teste)
y_treino = torch.LongTensor(y_treino)
y_teste = torch.LongTensor(y_teste)

#Precisamos instanciar a rede da classe Modelo, através do comando
modelo_classificacao = MinhaRede()

#Além de definir a rede neural, precisamos verificar se a rede está levando a entrada para um resultado 
#próximo da saída desejada. Nós fazemos isso através de uma função objetivo ou função de custo:
funcao_objetivo = nn.CrossEntropyLoss()

#Na primeira tentativa, a rede neural não irá obter uma saída satisfatória. Isso acontece porque os pesos 
#que ligam cada um dos neurônios são definidos de forma aleatória. Por isso, precisamos fazer a correção desses pesos. 
#esse processo é o mesmo que fizemos manualmene na Aula 04.02 nos valores de w1, w2 e b
#O otimizador que será utilizado nesse processo é definido por:
otimizador = torch.optim.Adam(modelo_classificacao.parameters(), lr=0.01)

#Finalmente vamos treinar a rede neural. O processo de propagação e retropropagação será repetido por 100 épocas. 
#Assim, esperamos corrigir os pesos para obter uma rede que transforma corretamente a entrada em uma previsão da classe da semente.
epocas = 100
custos = []
for i in range(epocas):
  y_predito = modelo_classificacao.forward(X_treino)
  custo = funcao_objetivo(y_predito, y_treino)
  custos.append(custo)

  otimizador.zero_grad()
  custo.backward()
  otimizador.step()

#Por fim, podemos tentar prever valores de y passando como entrada o X_teste. 
#Assim temos como comparar o Y e o YHat que é o valor estimado. 
#Aqui estão as dez primeiras linhas da minha tabela de resultado. 
#A última coluna da tabela retorna 1 para estimativas corretas e zero para as incorretas.
preds = []
with torch.no_grad():
    for val in X_teste:
        y_predito = modelo_classificacao.forward(val)
        preds.append(y_predito.argmax().item())
        df = pd.DataFrame({'Y': y_teste, 'YHat': preds})
        df['Correto'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
        df