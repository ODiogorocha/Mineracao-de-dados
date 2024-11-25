import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def carregar_dados(caminho_arquivo: str) -> pd.DataFrame:
    return pd.read_csv(caminho_arquivo)

def explorar_dados(dados: pd.DataFrame) -> None:
    
    print("Resumo estatistico:")
    print(dados.describe())
    print("\ninformacoes dos dados:")
    print(dados.info())

    #graficos basicos
    plt.figure(figsize=(12,6))
    sns.heatmap(dados.isnull(),cbar=False, cmap='viridis')
    plt.title("Valores ausentes")
    plt.show()

    dados.hist(bins=20, figsize=(12,10), color='skyblue', edgecolor='black')
    plt.subtitle("Distribuicao das variaveis", fontsize=16)
    plt.show()

def limpar_dados(dados: pd.DataFrame) -> pd.DataFrame:
    #trata valores duplicados e ausentes 

    dados = dados.drop_duplicates()
    dados = dados.fillna(dados.median())

    return dados

def transformar_dados(dados: pd.DataFrame, colunas_alvo: list) -> pd.DataFrame:
    #normalizar as colunas
    scaler = StandardScaler()
    dados[colunas_alvo] = scaler.fit_transform(dados[colunas_alvo])

    return dados

def reduzir_dimensao(dados: pd.DataFrame, n_componentes: int) -> pd.DataFrame:

    pca = PCA(n_components=n_componentes)
    dados_pca = pca.fit_transform(dados)
    print(f"Variancia explicada por componente: {pca.explained_variance_ratio_}")
    return pd.DataFrame(dados_pca)

def treinar_modelo(dados: pd.DataFrame, alvo: str) -> None:
    #divide dados treino e teste
    X = dados.drop(alvo, axis=1)
    y = dados[alvo]
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

    #treina modelo classificacao
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_treino, y_treino)

    previsoes = modelo.predict(X_teste)

    #avaliar modelo
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

    dados = carregar_dados(caminho_dados)
    explorar_dados(dados)

    dados_limpos = limpar_dados(dados)

    colunas_para_transformar = [col for col in dados_limpos.columns if dados_limpos[col].dtype != 'object']
    dados_transformados = transformar_dados(dados_limpos, colunas_para_transformar)

    dados_reduzidos = reduzir_dimensao(dados_limpos, colunas_para_transformar)

    coluna_alvo = "#FORMADOS"
    if coluna_alvo in dados_transformados.columns:
        treinar_modelo(dados_transformados, coluna_alvo)
    else:
        print(f"Coluna alvo '{coluna_alvo}' não encontrada nos dados!")



