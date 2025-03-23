import numpy as np
from sklearn.neighbors import kneighbors_graph

def getOrderbyRelevance(matriz_distancias):

   closeness = matriz_distancias.shape[0] / matriz_distancias.sum(axis=1) 

   R = np.argsort(closeness)

   return R

def getNearestPossibleNeighbor(v_i, K, Kmax, Knn, G, matriz_adjacencia):

   indices = np.where((Knn[v_i, :] != 0) & (matriz_adjacencia[v_i, :] == 0))[0]
   valores = Knn[v_i, indices]

   mapeamento = sorted(zip(indices, valores), key = lambda x : x[1]) 
   
   Kmax = min(Kmax, len(mapeamento))
   
   menor = None
   for i in range(Kmax):
      if G[mapeamento[i][0]] < K:
         menor = mapeamento[i][0]
         break

   return menor

def getNearestPossibleNeighborflex(v_i, Knn, G, matriz_adjacencia):

   indices = np.where((Knn[v_i, :] != 0) & (matriz_adjacencia[v_i, :] == 0))[0]
   valores = Knn[v_i, indices]

   mapeamento = sorted(zip(indices, valores), key = lambda x : x[1]) 
      
   menor = np.inf
   for i in range(len(mapeamento)):
      if G[mapeamento[i][0]] < menor:
         menor = mapeamento[i][0]

   return menor

def connect(P, v_i, v_j, G):

   P[v_i, v_j] = 1
   P[v_j, v_i] = 1
   G[v_i] += 1
   G[v_j] += 1

# SKNN para calcular a matriz de adjacencias
# entrada: matriz de distancia, k 
# saida: matriz de adjacencia
def Sknn(dados, matriz_distancias, medida_distancia, Kmax):
  
   N = dados.shape[0]
   matriz_adjacencia = np.zeros((N, N))
   G = np.zeros(N)

   # Converte em bool para usar a rogerstanimoto sem da warning
   if medida_distancia == 'rogerstanimoto':
     dados = dados.astype(bool)

   Knn =  kneighbors_graph(dados, Kmax, mode='connectivity',  metric = medida_distancia, include_self=False).toarray()
   Knn = np.maximum(Knn, Knn.T)
   R = getOrderbyRelevance(matriz_distancias)
  
   v_i = 0 
   v_j = 0
   K = 1
   while (K <= Kmax):
      for i in range(N):
         v_i = R[i]
         v_j = getNearestPossibleNeighbor(v_i, K, Kmax, Knn, G, matriz_adjacencia)
         if v_j != None:
            connect(matriz_adjacencia, v_i, v_j, G)
      K += 1

   while np.any( G < Kmax):
     indices = np.where(G < Kmax)[0]
     for i in range(len(indices)):
        v_i = indices[i]
        v_j = getNearestPossibleNeighborflex(v_i, Knn, G, matriz_adjacencia)
        if v_j != np.inf:
         connect(matriz_adjacencia, v_i, v_j, G)

   return matriz_adjacencia
