import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from spellchecker import SpellChecker

# Carregar o arquivo CSV
caminho_arquivo = "Mineracao-de-dados/Trabalho-1/padaria.csv"
dados = pd.read_csv(caminho_arquivo)

# Inicializando o corretor ortográfico
spell = SpellChecker(language='pt')

# Função para verificar erros de português
def verificar_portugues(texto):
    palavras_desconhecidas = []
    for palavra in texto.split(','):
        if spell.unknown([palavra.strip()]):
            palavras_desconhecidas.append(palavra.strip())
    return palavras_desconhecidas

# Função para corrigir erros de português
def corrigir_produtos(produtos):
    produtos_corrigidos = []
    for produto in produtos.split(','):
        produto = produto.strip()
        # Sugestão da palavra correta
        palavra_corrigida = spell.candidates(produto)
        if palavra_corrigida:
            produtos_corrigidos.append(next(iter(palavra_corrigida)))  # pega a primeira sugestão
        else:
            produtos_corrigidos.append(produto)  # se não houver sugestão, mantém a palavra original
    return ', '.join(produtos_corrigidos)

# Aplicar a função para encontrar produtos com erro
erros = dados['produtos'].apply(verificar_portugues)
produtos_com_erro = erros[erros.apply(lambda x: len(x) > 0)]

# Imprimir produtos com erro
print("\nProdutos com erros:")
print(produtos_com_erro)

# Corrigir os produtos
dados['produtos'] = dados['produtos'].apply(corrigir_produtos)

# Filtrando dados para excluir linhas com erros
dados_filtrados = dados[~dados['produtos'].apply(lambda x: len(verificar_portugues(x)) > 0)]
print("Dados Filtrados:")
print(dados_filtrados.head())

# Verificar se dados_filtrados está vazio
if dados_filtrados.empty:
    print("\nNão há dados válidos para análise após a filtragem. Verifique os produtos com erros.")
    exit()

# Criando um DataFrame de transações
transacoes = dados_filtrados.groupby('compra')['produtos'].apply(lambda x: ','.join(x)).reset_index()

# Transformando produtos em formato one-hot
produtos_dummies = transacoes['produtos'].str.get_dummies(sep=', ')
transacoes = transacoes.drop(columns='produtos').join(produtos_dummies)

# Aplicando o Apriori
frequent_itemsets = apriori(transacoes, min_support=0.01, use_colnames=True)

# Gerar regras de associação
regras = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Verificar se há regras geradas
if regras.empty:
    print("\nNenhuma regra de associação encontrada.")
    exit()

# Filtrar regras 1 para 1 (um antecedente e um consequente)
regras_1_para_1 = regras[(regras['antecedents'].apply(lambda x: len(x) == 1)) & (regras['consequents'].apply(lambda x: len(x) == 1))]

# Ordenar as regras 1 para 1 por confiança
regras_1_para_1 = regras_1_para_1.sort_values(by='confidence', ascending=False).head(5)

# Exibir as regras 1 para 1
print("\nRegras 1 para 1 (prodA => prodB):")
print(regras_1_para_1)

# Exportar regras 1 para 1 para um arquivo CSV
regras_1_para_1.to_csv("regras_1_para_1.csv", index=False)

# Finalizando
print("\nProcesso completo! As regras 1 para 1 foram exportadas para 'regras_1_para_1.csv'.")
