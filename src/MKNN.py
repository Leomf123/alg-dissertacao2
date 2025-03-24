import numpy as np
from sklearn.neighbors import kneighbors_graph
from prim import primMST

def adicionarMST(matriz_adjacencia, matriz_distancia):

    MaxST = np.array(primMST(matriz_distancia))
    MaxST[MaxST != 0] = 1

    # Mesclar as matrizes
    matriz = np.where((matriz_adjacencia == 1) | (MaxST == 1), 1, 0)

    return matriz

# MKNN para calcular a matriz de adjacencias
# entrada: matriz de distancia, k
# saida: matriz de adjacencia
def Mknn(dados, matriz_distancia, medida_distancia, k):

    # Converte em bool para usar a rogerstanimoto sem da warning
    if medida_distancia == 'rogerstanimoto':
        dados = dados.astype(bool)

    matriz_adjacencia =  kneighbors_graph(dados, k, mode='connectivity',  metric = medida_distancia, include_self=False).toarray()

    matriz_adjacencia_transposta = matriz_adjacencia.T

    matriz_adjacencia = np.minimum(matriz_adjacencia, matriz_adjacencia_transposta)

    matriz_adjacencia = adicionarMST(matriz_adjacencia, matriz_distancia)

    return matriz_adjacencia
