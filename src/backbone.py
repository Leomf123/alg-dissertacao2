import numpy as np

def backbone(matriz_pesos, alpha):

    N = matriz_pesos.shape[0]

    matriz_backbone = np.copy(matriz_pesos)
    matriz_somas = matriz_pesos.sum(axis=1)

    matriz_adjacencia = (matriz_pesos > 0).astype(int)
    graus = matriz_adjacencia.sum(axis=1)

    for i in range(N):
        for j in range(i + 1, N):

            K_i = graus[i]
            K_j = graus[j]
            p_i = (1 - (matriz_pesos[i, j] / matriz_somas[i])) ** (K_i - 1)
            p_j = (1 - (matriz_pesos[j, i] / matriz_somas[j])) ** (K_j - 1)

            p = min(p_i, p_j)

            if p >= alpha:
                matriz_backbone[i, j] = 0
                matriz_backbone[j, i] = 0
    
    return matriz_backbone


def retonarBackBone(matriz_pesos, alpha, esparcidade):

    if esparcidade == "Sim":
        return backbone(matriz_pesos, alpha)
    else:
        return matriz_pesos