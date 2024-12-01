import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def carregar_dados(caminho_arquivo: str) -> pd.DataFrame:
    try:
        if caminho_arquivo.endswith(('.xls', '/home/diogo/Documents/Aulas/Mineracao-de-dados/Projetos/Projeto2/arq/Ingressa.xlsx')):
            return pd.read_excel(caminho_arquivo)
        elif caminho_arquivo.endswith('.csv'):
            return pd.read_csv(caminho_arquivo)
        else:
            raise ValueError("Formato de arquivo não suportado! Use .csv, .xls ou .xlsx.")
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        raise

def explorar_dados(dados: pd.DataFrame) -> None:
    print("Resumo estatístico:")
    print(dados.describe())
    print("\nInformações dos dados:")
    print(dados.info())

    # Gráficos básicos
    plt.figure(figsize=(12, 6))
    sns.heatmap(dados.isnull(), cbar=False, cmap='viridis')
    plt.title("Valores ausentes")
    plt.show()

    dados.hist(bins=20, figsize=(12, 10), color='skyblue', edgecolor='black')
    plt.suptitle("Distribuição das variáveis", fontsize=16)
    plt.show()

def limpar_dados(dados: pd.DataFrame) -> pd.DataFrame:
    # Trata valores duplicados e ausentes
    dados = dados.drop_duplicates()
    dados = dados.fillna(dados.median(numeric_only=True))

    return dados

def transformar_dados(dados: pd.DataFrame) -> pd.DataFrame:
    # Seleciona apenas as colunas numéricas para a normalização
    colunas_numericas = dados.select_dtypes(include=['number']).columns
    if not colunas_numericas.empty:
        scaler = StandardScaler()
        dados[colunas_numericas] = scaler.fit_transform(dados[colunas_numericas])

    return dados

def reduzir_dimensao(dados: pd.DataFrame, n_componentes: int) -> pd.DataFrame:
    pca = PCA(n_components=n_componentes)
    dados_pca = pca.fit_transform(dados)
    print(f"Variância explicada por componente: {pca.explained_variance_ratio_}")
    return pd.DataFrame(dados_pca)

def treinar_modelo(dados: pd.DataFrame, alvo: str) -> None:
    # Divide dados treino e teste
    X = dados.drop(alvo, axis=1)
    y = dados[alvo]
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treina modelo de classificação
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_treino, y_treino)

    previsoes = modelo.predict(X_teste)

    # Avaliar modelo
    print("Relatório de classificação:")
    print(classification_report(y_teste, previsoes))
    print("\nMatriz de confusão:")
    sns.heatmap(confusion_matrix(y_teste, previsoes), annot=True, cmap='Blues', fmt='d')
    plt.title("Matriz de confusão")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.show()

def aplicar_apriori(dados: pd.DataFrame, suporte_minimo: float = 0.3, confiança_minima: float = 0.6):
    # Transformar dados em codificação binária
    dados_codificados = pd.get_dummies(dados, drop_first=True)

    # Encontrar conjuntos frequentes
    conjuntos_frequentes = apriori(dados_codificados, min_support=suporte_minimo, use_colnames=True)
    print("\nConjuntos frequentes:")
    print(conjuntos_frequentes)

    # Gerar regras de associação
    regras = association_rules(conjuntos_frequentes, metric='lift', min_threshold=confiança_minima)
    print("\nRegras de associação:")
    print(regras)

if __name__ == "__main__":
    caminho_dados = "Projetos/Projeto2/arq/Ingressantes e Formandos/CT Ingressantes e formados por sexo.xls"

    try:
        dados = carregar_dados(caminho_dados)
        explorar_dados(dados)

        dados_limpos = limpar_dados(dados)

        dados_transformados = transformar_dados(dados_limpos)

        dados_reduzidos = reduzir_dimensao(dados_transformados, n_componentes=2)

        coluna_alvo = "NOME_UNIDADE"  
        if coluna_alvo in dados_transformados.columns:
            print("Escolha a técnica para a análise:")
            print("1. Random Forest Classifier")
            print("2. Apriori")
            escolha = input("Digite o número da técnica desejada: ")

            if escolha == '1':
                treinar_modelo(dados_transformados, coluna_alvo)
            elif escolha == '2':
                aplicar_apriori(dados_transformados)
            else:
                print("Escolha inválida!")
        else:
            print(f"Coluna alvo '{coluna_alvo}' não encontrada nos dados!")
    except Exception as e:
        print(f"Erro durante a execução: {e}")
