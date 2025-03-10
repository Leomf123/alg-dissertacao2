import heapq

def primMST(grafo):
    V = len(grafo)  
    pai = [-1] * V  
    chave = [float('inf')] * V  
    V_bool = [False] * V 

    chave[0] = 0
    min_heap = [(0, 0)] 

    while min_heap:
        
        _, u = heapq.heappop(min_heap)
        V_bool[u] = True

        
        for v in range(V):
            
            if grafo[u][v] > 0 and not V_bool[v] and chave[v] > grafo[u][v]:
                chave[v] = grafo[u][v]
                pai[v] = u
                heapq.heappush(min_heap, (chave[v], v))

    
    MST = [[0] * V for _ in range(V)]
    for v in range(1, V):
        u = pai[v]
        MST[u][v] = grafo[u][v]
        MST[v][u] = grafo[u][v]

    return MST

# Construção da Árvore Geradora Máxima
def primMaxST(grafo):
    V = len(grafo)  
    pai = [-1] * V  
    chave = [-float('inf')] * V 
    V_bool = [False] * V  

    chave[0] = 0
    max_heap = [(0, 0)]
    while max_heap:
        _, u = heapq.heappop(max_heap)
        V_bool[u] = True

        for v in range(V):
            if grafo[u][v] > 0 and not V_bool[v] and chave[v] < grafo[u][v]:
                chave[v] = grafo[u][v]
                pai[v] = u
                heapq.heappush(max_heap, (-chave[v], v))

    MaxST = [[0] * V for _ in range(V)]
    for v in range(1, V):
        u = pai[v]
        MaxST[u][v] = grafo[u][v]
        MaxST[v][u] = grafo[u][v]

    return MaxST

