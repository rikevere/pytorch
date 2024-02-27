import torch

# Cria um tensor aleatório tns1 com dimensionalidade 7 x 7 x 3
tns1 = torch.rand(7, 7, 3)

# Cria outro tensor aleatório tns2 de 147 x 1
tns2 = torch.rand(147, 1)

# Modifica tns1 para ter a mesma forma que tns2
# Primeiro, achata tns1 para uma dimensão única de 147 elementos
# Depois, usa view para ajustar a forma para 147 x 1
tns1_modificado = tns1.view(-1, 1)

# Soma os dois tensores
resultado = tns1_modificado + tns2

print("Resultado da soma:", resultado)
print("Forma do tensor resultante:", resultado.shape)
