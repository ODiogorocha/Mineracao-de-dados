import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Funções de processamento e análise para Apriori
def carregar_dados(caminho_arquivo: str) -> pd.DataFrame:
    try:
        if caminho_arquivo.endswith(('.xls', '.xlsx')):
            return pd.read_excel(caminho_arquivo)
        elif caminho_arquivo.endswith('.csv'):
            return pd.read_csv(caminho_arquivo)
        else:
            raise ValueError("Formato de arquivo não suportado! Use .csv, .xls ou .xlsx.")
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        raise

def verificar_e_transformar_dados(dados: pd.DataFrame) -> pd.DataFrame:
    print("Colunas disponíveis no dataset:", dados.columns.tolist())
    print("\nExemplo de dados:")
    print(dados.head())

    try:
        cesta = dados.apply(lambda x: x.astype(str))
        cesta_transformed = pd.get_dummies(cesta, columns=dados.columns)
        print("\nDados transformados em matriz de cestas:")
        print(cesta_transformed.head())
        return cesta_transformed
    except Exception as e:
        print(f"Erro ao transformar os dados para cestas: {e}")
        raise

def aplicar_apriori(cesta: pd.DataFrame, suporte_minimo: float = 0.01) -> pd.DataFrame:
    try:
        conjuntos_frequentes = apriori(cesta, min_support=suporte_minimo, use_colnames=True)
        print("\nConjuntos frequentes encontrados:")
        print(conjuntos_frequentes)
        return conjuntos_frequentes
    except Exception as e:
        print(f"Erro ao aplicar o Apriori: {e}")
        raise

def gerar_regras(conjuntos_frequentes: pd.DataFrame, num_itemsets: int, metric: str = 'confidence', min_threshold: float = 0.5) -> pd.DataFrame:
    try:
        regras = association_rules(conjuntos_frequentes, metric=metric, min_threshold=min_threshold, num_itemsets=num_itemsets)
        print("\nRegras de associação encontradas:")
        print(regras)
        return regras
    except Exception as e:
        print(f"Erro ao gerar regras: {e}")
        raise

def gerar_grafico_comparacao_sexo(dados: pd.DataFrame):
    try:
        comparacao_sexo = dados.groupby('SEXO')['FORMADOS'].sum()

        cores = {'M': 'blue', 'F': 'pink'}  
        comparacao_sexo.plot(kind='bar', title='Comparação de Formandos por Sexo', color=[cores[x] for x in comparacao_sexo.index])

        plt.xlabel('Sexo')
        plt.ylabel('Número de Formandos')
        plt.show()
    except Exception as e:
        print(f"Erro ao gerar o gráfico de comparação: {e}")
        raise


# Funções de processamento e análise para a árvore de decisão
def explorar_dados(dados: pd.DataFrame) -> None:
    print("Resumo estatístico:")
    print(dados.describe(include='all'))
    print("\nInformações dos dados:")
    print(dados.info())

    plt.figure(figsize=(12, 6))
    sns.heatmap(dados.isnull(), cbar=False, cmap='viridis')
    plt.title("Valores ausentes")
    plt.show()

    dados.hist(bins=20, figsize=(12, 10), color='skyblue', edgecolor='black')
    plt.suptitle("Distribuição das variáveis", fontsize=16)
    plt.show()

def limpar_dados(dados: pd.DataFrame) -> pd.DataFrame:
    dados = dados.drop_duplicates()
    dados = dados.fillna(dados.median(numeric_only=True))
    return dados

def transformar_dados(dados: pd.DataFrame) -> pd.DataFrame:
    dados_numericos = dados.select_dtypes(include=['float64', 'int64'])
    dados_categoricos = dados.select_dtypes(include=['object'])

    scaler = StandardScaler()
    dados_numericos = pd.DataFrame(scaler.fit_transform(dados_numericos), columns=dados_numericos.columns)

    for col in dados_categoricos.columns:
        le = LabelEncoder()
        dados_categoricos[col] = le.fit_transform(dados_categoricos[col])

    dados_transformados = pd.concat([dados_numericos, dados_categoricos], axis=1)
    return dados_transformados

def reduzir_dimensao(dados: pd.DataFrame, n_componentes: int) -> pd.DataFrame:
    pca = PCA(n_components=n_componentes)
    dados_pca = pca.fit_transform(dados)
    print(f"Variância explicada por componente: {pca.explained_variance_ratio_}")
    return pd.DataFrame(dados_pca)

def treinar_modelo(dados: pd.DataFrame, alvo: str) -> None:
    X = dados.drop(alvo, axis=1)
    y = dados[alvo]
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_treino, y_treino)

    previsoes = modelo.predict(X_teste)

    print("Relatório de classificação:")
    print(classification_report(y_teste, previsoes))
    print("\nMatriz de confusão:")
    sns.heatmap(confusion_matrix(y_teste, previsoes), annot=True, cmap='Blues', fmt='d')
    plt.title("Matriz de confusão")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.show()

    # Gráfico de comparação por sexo
    dados_teste = X_teste.copy()
    dados_teste['SEXO'] = y_teste
    dados_teste['Predicao'] = previsoes
    comparacao_sexo = dados_teste.groupby('SEXO').agg({'Predicao': 'value_counts'}).unstack(fill_value=0)
    comparacao_sexo.plot(kind='bar', stacked=True, figsize=(12, 6), title='Comparação de Predições por Sexo')
    plt.xlabel('Sexo')
    plt.ylabel('Número de Predições')
    plt.show()

    # Gráfico de pizza para a distribuição de previsões por sexo
    distribuicao_sexo = dados_teste['SEXO'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(distribuicao_sexo, labels=distribuicao_sexo.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightcoral'])
    plt.title("Distribuição de Previsões por Sexo")
    plt.show()

# Função principal
def main():
    caminho_dados = "Projetos/Projeto2/arq/Ingressantes e Formandos/CT Ingressantes e formados por sexo.xls"

    print("Escolha o método desejado:")
    print(" (1) Árvore de Decisão")
    print(" (2) Apriori")
    try:
        metodo = int(input("Digite o número do método desejado: "))
        
        if metodo == 1:
            try:
                dados = carregar_dados(caminho_dados)
                explorar_dados(dados)
                dados_limpos = limpar_dados(dados)
                dados_transformados = transformar_dados(dados_limpos)
                dados_reduzidos = reduzir_dimensao(dados_transformados, n_componentes=2)

                coluna_alvo = "NOME_UNIDADE"
                if coluna_alvo in dados_transformados.columns:
                    treinar_modelo(dados_transformados, coluna_alvo)
                else:
                    print(f"Coluna alvo '{coluna_alvo}' não encontrada nos dados!")
            except Exception as e:
                print(f"Erro durante a execução da árvore de decisão: {e}")

        elif metodo == 2:
            print("Iniciando Apriori...")
            try:
                dados = carregar_dados(caminho_dados)
                cesta = verificar_e_transformar_dados(dados)
                suporte_minimo = 0.02
                conjuntos_frequentes = aplicar_apriori(cesta, suporte_minimo)
                num_itemsets = len(conjuntos_frequentes)
                regras = gerar_regras(conjuntos_frequentes, num_itemsets, metric='lift', min_threshold=1.0)
                gerar_grafico_comparacao_sexo(dados)
            except Exception as e:
                print(f"Erro durante a execução do Apriori: {e}")

        else:
            print("Opção inválida! Por favor, escolha 1 ou 2.")
    except ValueError:
        print("Entrada inválida! Por favor, insira apenas números.")

if __name__ == "__main__":
    main()
