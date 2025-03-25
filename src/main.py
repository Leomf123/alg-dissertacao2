import sys
from teste import teste


def main(indice_dataset):
    
    datasets = [
        "armstrong2002v1.data",
        "chowdary2006.data",
        "ace_ECFP_4.data",
        "ace_ECFP_6.data",
        "cox2_ECFP_6.data",
        "dhfr_ECFP_4.data",
        "dhfr_ECFP_6.data",
        "fontaine_ECFP_4.data",
        "fontaine_ECFP_6.data",
        "m1_ECFP_4.data",
        "m1_ECFP_6.data",
        "transplant.data",
        "autoPrice.data",
        "seeds.data",
        "chscase_geyser1.data",
        "diggle_table.data",
        "gordon2002.data",
        "articles_1442_5.data",
        "articles_1442_80.data",
        "iris.data",
        "analcatdata_authorship-458.data",
        "wine-187.data",
        "banknote-authentication.data",
        "yeast_Galactose.data",
        "mfeat-karhunen.data",
        "mfeat-factors.data",
        "semeion.data",
        "wdbc.data",
        "stock.data",
        "segmentation-normcols.data",
        "cardiotocography.data",
    ]

    K = [4, 6, 8, 10, 12, 14, 16]

    Adjacencia = ["MST", "mutKNN", "symKNN", "symFKNN", "SKNN", "MKNN"]

    Ponderacao = ["RBF", "HM", "LLE"]

    Esparcidade = ["Nao", "Sim"] #Essa ordem importa no código de teste

    Quantidade_rotulos = [0.02, 0.05, 0.08, 0.1]

    Quantidade_experimentos = 20
    
    teste(indice_dataset, datasets, K, Adjacencia, Ponderacao, Quantidade_rotulos, Quantidade_experimentos, Esparcidade)


if __name__ == "__main__":

    indice_dataset = int(sys.argv[1])

    main(indice_dataset)
