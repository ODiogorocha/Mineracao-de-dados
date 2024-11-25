# Documentação do Código de Análise e Mineração de Dados

Este documento detalha cada parte do código, explicando suas funções e as bibliotecas utilizadas. O objetivo do código é realizar análise, limpeza, transformação e aplicação de aprendizado de máquina em um dataset relacionado à UFSM.

---

## Bibliotecas Utilizadas

### 1. **Pandas**
- **Descrição:** Utilizada para manipulação e análise de dados estruturados.
- **Funções Usadas:** 
  - `read_excel`: Para carregar arquivos Excel (`.xls` ou `.xlsx`).
  - `read_csv`: Para carregar arquivos CSV.
  - `describe`: Para obter estatísticas descritivas.
  - `info`: Para exibir informações sobre o DataFrame.
  - `drop_duplicates`: Para remover linhas duplicadas.
  - `fillna`: Para preencher valores ausentes.

### 2. **Matplotlib**
- **Descrição:** Biblioteca para criação de visualizações gráficas.
- **Funções Usadas:**
  - `plt.figure`: Para ajustar o tamanho da figura.
  - `plt.title`: Para adicionar títulos aos gráficos.
  - `plt.show`: Para exibir os gráficos.

### 3. **Seaborn**
- **Descrição:** Biblioteca para visualização de dados, construída sobre o Matplotlib.
- **Funções Usadas:**
  - `sns.heatmap`: Para visualizar valores ausentes ou matrizes de confusão.

### 4. **Scikit-learn**
- **Descrição:** Biblioteca para aprendizado de máquina e mineração de dados.
- **Funções Usadas:**
  - `StandardScaler`: Para normalização dos dados.
  - `PCA`: Para redução de dimensionalidade.
  - `train_test_split`: Para dividir os dados em conjuntos de treino e teste.
  - `RandomForestClassifier`: Para treinar um modelo de classificação.
  - `classification_report`: Para gerar métricas de desempenho do modelo.
  - `confusion_matrix`: Para gerar uma matriz de confusão.

---

## Funções do Código

### 1. `carregar_dados(caminho_arquivo: str) -> pd.DataFrame`
- **Objetivo:** Carregar dados de um arquivo Excel ou CSV.
- **Parâmetros:**
  - `caminho_arquivo`: Caminho para o arquivo a ser carregado.
- **Retorno:** Um `DataFrame` com os dados carregados.
- **Detalhes:**
  - Verifica a extensão do arquivo.
  - Usa `pd.read_excel` para arquivos Excel e `pd.read_csv` para arquivos CSV.
  - Gera um erro se o formato do arquivo não for suportado.

---

### 2. `explorar_dados(dados: pd.DataFrame) -> None`
- **Objetivo:** Explorar os dados carregados, exibindo estatísticas básicas e visualizações.
- **Parâmetros:**
  - `dados`: O DataFrame contendo os dados.
- **Processo:**
  - Exibe estatísticas descritivas com `describe`.
  - Exibe informações sobre os dados (tipos de coluna, valores nulos) com `info`.
  - Gera:
    - Um gráfico de calor para mostrar valores ausentes.
    - Histogramas para mostrar a distribuição das variáveis.

---

### 3. `limpar_dados(dados: pd.DataFrame) -> pd.DataFrame`
- **Objetivo:** Realizar limpeza básica dos dados.
- **Parâmetros:**
  - `dados`: O DataFrame a ser limpo.
- **Processo:**
  - Remove linhas duplicadas com `drop_duplicates`.
  - Preenche valores ausentes com a mediana das colunas numéricas usando `fillna`.

---

### 4. `transformar_dados(dados: pd.DataFrame, colunas_alvo: list) -> pd.DataFrame`
- **Objetivo:** Normalizar as colunas especificadas.
- **Parâmetros:**
  - `dados`: O DataFrame contendo os dados.
  - `colunas_alvo`: Lista de colunas que devem ser normalizadas.
- **Processo:**
  - Normaliza as colunas especificadas usando `StandardScaler`.

---

### 5. `reduzir_dimensao(dados: pd.DataFrame, n_componentes: int) -> pd.DataFrame`
- **Objetivo:** Reduzir a dimensionalidade dos dados.
- **Parâmetros:**
  - `dados`: O DataFrame contendo os dados.
  - `n_componentes`: Número de componentes principais a serem mantidos.
- **Processo:**
  - Aplica PCA para reduzir a dimensionalidade.
  - Exibe a variância explicada por cada componente.

---

### 6. `treinar_modelo(dados: pd.DataFrame, alvo: str) -> None`
- **Objetivo:** Treinar um modelo de classificação e avaliar o desempenho.
- **Parâmetros:**
  - `dados`: O DataFrame contendo os dados.
  - `alvo`: Nome da coluna alvo (variável dependente).
- **Processo:**
  - Divide os dados em treino e teste com `train_test_split`.
  - Treina um modelo de floresta aleatória com `RandomForestClassifier`.
  - Avalia o modelo usando:
    - Relatório de classificação (`classification_report`).
    - Matriz de confusão (`confusion_matrix`).

---

### Bloco `if __name__ == "__main__":`
- **Objetivo:** Executar o fluxo completo de análise.
- **Etapas:**
  1. Carregar os dados do arquivo especificado.
  2. Explorar os dados para entender suas características.
  3. Limpar os dados para remover duplicatas e preencher valores ausentes.
  4. Transformar os dados para normalizar colunas relevantes.
  5. Reduzir a dimensionalidade usando PCA.
  6. Treinar e avaliar um modelo de classificação com a coluna alvo `#FORMADOS`.

---

## Ajustes Possíveis
- **Coluna Alvo:** Certifique-se de que o nome da coluna alvo (`#FORMADOS`) corresponde ao presente no dataset.
- **Caminho do Arquivo:** Verifique se o caminho especificado está correto.

---

## Exemplo de Uso

1. Coloque o dataset no caminho especificado no código.
2. Execute o script.
3. Os gráficos e relatórios gerados serão exibidos diretamente no console e nas janelas gráficas.

---
