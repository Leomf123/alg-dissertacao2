import pandas as pd

def gravar_resultados(indice_dataset, test_ID, nome_dataset, k, adjacencia, simetrica, conectado, positivo, ponderacao, r, e, esparcidade, seed, nRotulos, acuracia, f_measure):
    
    if test_ID == 0: 

        # Criando um dataframe
        dados = [{'test_ID': test_ID, 'Dataset': nome_dataset, 'Adjacencia': adjacencia, 'k': k, 'Ponderacao': ponderacao, 'Simetrica': simetrica, 'Conectado': conectado, 'Positivo': positivo, 'Esparc': esparcidade, 'PorcRot': r, 'NumExp': e, 'SeedExp': seed, 'NumNRot': nRotulos, 'Acuracia': acuracia, 'F_measure': f_measure}]

        df = pd.DataFrame(dados)
        # salvo arquivo csv
        df.to_csv('Resultados' + str(indice_dataset) + '.csv', index=False)

    else:
        
        # leio arquivo csv existente e salvo df
        df = pd.read_csv('Resultados' + str(indice_dataset) + '.csv')
  
        # Adicionando dados
        dados = [{'test_ID': test_ID, 'Dataset': nome_dataset, 'Adjacencia': adjacencia, 'k': k, 'Ponderacao': ponderacao, 'Simetrica': simetrica, 'Conectado': conectado, 'Positivo': positivo, 'Esparc': esparcidade, 'PorcRot': r, 'NumExp': e, 'SeedExp': seed, 'NumNRot': nRotulos, 'Acuracia': acuracia, 'F_measure': f_measure}]

        dados = pd.DataFrame(dados)
        df = pd.concat([df, dados], ignore_index=True)

        # salvo arquivo csv mesmo lugar do outro
        df.to_csv('Resultados' + str(indice_dataset) + '.csv', index=False)