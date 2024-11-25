import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

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

def explorar_dados(dados: pd.DataFrame) -> None:
    print("Resumo estatistico:")
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

def transformar_dados(dados: pd.DataFrame, colunas_alvo: list) -> pd.DataFrame:
    # Normaliza as colunas
    scaler = StandardScaler()
    dados[colunas_alvo] = scaler.fit_transform(dados[colunas_alvo])

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

if __name__ == "__main__":
    caminho_dados = "Projetos/Projeto2/arq/Ingressantes e Formandos/CT Ingressantes e formados por sexo.xls"

    try:
        dados = carregar_dados(caminho_dados)
        explorar_dados(dados)

        dados_limpos = limpar_dados(dados)

        colunas_para_transformar = [col for col in dados_limpos.columns if dados_limpos[col].dtype != 'object']
        dados_transformados = transformar_dados(dados_limpos, colunas_para_transformar)

        dados_reduzidos = reduzir_dimensao(dados_transformados, n_componentes=2)

        coluna_alvo = "#FORMADOS"  
        if coluna_alvo in dados_transformados.columns:
            treinar_modelo(dados_transformados, coluna_alvo)
        else:
            print(f"Coluna alvo '{coluna_alvo}' não encontrada nos dados!")
    except Exception as e:
        print(f"Erro durante a execução: {e}")
