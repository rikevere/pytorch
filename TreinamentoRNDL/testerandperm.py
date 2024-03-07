import torch

# Suponha que você tenha 10 dados e queira embaralhá-los
n = 10

# Gera uma permutação aleatória de índices de 0 a n-1
indices = torch.randperm(n)

print("Índices embaralhados:", indices)

# Exemplo de como usar os índices para embaralhar um tensor de dados
dados = torch.arange(1, n+1)  # Cria um tensor com valores de 1 a 10
dados_embaralhados = dados[indices]

print("Dados originais:", dados)
print("Dados embaralhados:", dados_embaralhados)
