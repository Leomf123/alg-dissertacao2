from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

def gerar_matriz_distancias(X, Y, medida_distancia ):

  matriz = cdist(X, Y, medida_distancia )

  return matriz

def definir_medida_distancia(nome_dado):

    datasets_euclidean = [
        "autoPrice.data",
        "banknote-authentication.data",
        "chscase_geyser1.data",
        "diggle_table.data",
        "iris.data",
        "seeds.data",
        "segmentation-normcols.data",
        "stock.data",
        "transplant.data",
        "wdbc.data",
        "wine-187.data",
        "yeast_Galactose.data",
        "mfeat-factors.data",
        "mfeat-karhunen.data",
        "cardiotocography.data"
    ]    
    datasets_tanimoto = [
        "ace_ECFP_4.data",
        "ace_ECFP_6.data",
        "cox2_ECFP_6.data",
        "dhfr_ECFP_4.data",
        "dhfr_ECFP_6.data",
        "fontaine_ECFP_4.data",
        "fontaine_ECFP_6.data",
        "m1_ECFP_4.data",
        "m1_ECFP_6.data",
    ]

    datasets_cosine = [
        "articles_1442_5.data",
        "articles_1442_80.data",
        "analcatdata_authorship-458.data",
        "armstrong2002v1.data",
        "chowdary2006.data",
        "gordon2002.data",
        "semeion.data"
    ]

    if nome_dado in datasets_tanimoto:
        return 'rogerstanimoto'
    elif nome_dado in datasets_cosine:
        return  'cosine'
    else:
        return 'euclidean'
    
def normalizar_dados(nome_dado, dados):

    datasets_Normalizar = [
        "autoPrice.data",
        "banknote-authentication.data",
        "stock.data",
        "transplant.data"
    ]    

    dados_normalizados = np.array(dados)

    if nome_dado in datasets_Normalizar:
        mean = np.mean(dados, axis=0)
        std = np.std(dados, axis=0)
        dados_normalizados = (dados - mean) / std
    
    return dados_normalizados

def checar_matrix_adjacencias(matriz_adjacencias):
    # Checar se a matriz é simétrica
    simetrica = np.array_equal(matriz_adjacencias, matriz_adjacencias.T)
    
    # Checar conectividade: nenhuma linha deve ter soma 0
    conectado = not np.any(np.sum(matriz_adjacencias, axis=1) == 0)
    
    # Checar se todos os valores são positivos ou zero
    positivo = not np.any(matriz_adjacencias < 0)
    
    return simetrica, conectado, positivo

def retornar_sigma(matriz_distancias, k):
    
    sigma = np.sort(matriz_distancias, axis=1)[:, k]
    sigma = sigma.sum()

    return sigma / (3 * matriz_distancias.shape[0])

def ordem_rotulos_primeiro(rotulos):

    # Máscara para identificar elementos rotulados
    mascara_rotulos = rotulos != 0

    # Índices de rótulos e não rótulos
    posicoes_rotulos = np.flatnonzero(mascara_rotulos)
    nao_rotulados = np.flatnonzero(~mascara_rotulos)

    # Combina diretamente os índices
    ordemObjetos = np.hstack((posicoes_rotulos, nao_rotulados))

    return posicoes_rotulos, ordemObjetos

def laplacianas(matriz_pesos):

    n = matriz_pesos.shape[0]
    
    # Matriz diagonal (grau dos vértices)
    D = np.diag(np.sum(matriz_pesos, axis=1))
        
    # Laplaciana normalizada
    epsilon = 1e-10  # Pequeno valor para estabilidade numérica
    D_diag = np.diag(D) + epsilon  # Evita divisão por zero
    D_inv_raiz = np.diag(1 / np.sqrt(D_diag))
    L_normalizada = 1.01 * np.eye(n) - D_inv_raiz @ matriz_pesos @ D_inv_raiz

    return L_normalizada
