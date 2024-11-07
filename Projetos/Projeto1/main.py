import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Função para carregar dados do CSV
def carregar_dados(arquivo_csv):
    # Lendo o arquivo CSV
    dados = pd.read_csv(arquivo_csv, header=None)
    
    # Separar os produtos em cada compra
    dados[1] = dados[1].str.split(", ")
    
    # Converter os dados em um formato transacional
    transacoes = dados[1].explode().reset_index(drop=True)
    
    # Criar um DataFrame com a contagem de ocorrências
    transacoes_dummies = pd.get_dummies(transacoes).groupby(level=0).sum()
    
    return transacoes_dummies

# Função para encontrar regras de associação
def encontrar_regras_associacao(transacoes, suporte_minimo=0.05, confianca_minima=0.5):
    # Encontrar itemsets frequentes
    itemsets_frequentes = apriori(transacoes, min_support=suporte_minimo, use_colnames=True)
    
    # Gerar regras de associação
    regras = association_rules(itemsets_frequentes, metric="confidence", min_threshold=confianca_minima)
    
    return regras

# Função principal
def main():
    arquivo_csv = "/home/diogo/Documents/Aulas/M.dados/Mineracao-de-dados/Trabalho-1/padaria.csv"  
    
    # Carregar dados
    transacoes = carregar_dados(arquivo_csv)
    
    # Encontrar regras de associação
    regras = encontrar_regras_associacao(transacoes)
    
    # Exibir todas as regras
    print("Todas as regras de associação:")
    if not regras.empty:
        print(regras)
    else:
        print("Nenhuma regra encontrada.")
    
    # Verifique Doce
    if "Doce" in transacoes.columns:
        print("\n'Doce' está presente nas transações.")
    else:
        print("\n'Doce' não está presente nas transações. Verifique os dados.")
    
    # Regras que implicam a compra de Doce
    regras_doce = regras[regras['consequents'].apply(lambda x: 'Doce' in x)]
    print("\nRegras que implicam a compra de 'Doce':")
    
    if not regras_doce.empty:
        print(regras_doce)
    else:
        print("Nenhuma regra que implica a compra de 'Doce' encontrada.")

# Executar o script
if __name__ == "__main__":
    main()
