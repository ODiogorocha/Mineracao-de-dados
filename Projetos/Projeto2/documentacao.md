# Documentação do Código

Este código implementa um processo de análise de dados usando duas abordagens: **Árvore de Decisão** e **Apriori**. O código inclui funções para carregar e processar dados, aplicar modelos de aprendizado de máquina e gerar gráficos para análise visual.

## Bibliotecas Utilizadas

- **pandas**: Manipulação e análise de dados.
- **matplotlib.pyplot**: Geração de gráficos.
- **seaborn**: Visualização estatística.
- **mlxtend.frequent_patterns**: Implementação do algoritmo Apriori para análise de associações.
- **sklearn**: Biblioteca para aprendizado de máquina, incluindo escalonamento de dados, modelagem e avaliação.

## Funções

### 1. `carregar_dados(caminho_arquivo: str) -> pd.DataFrame`
Carrega os dados de um arquivo Excel ou CSV.

- **Entrada**: Caminho do arquivo.
- **Saída**: DataFrame contendo os dados carregados.

### 2. `verificar_e_transformar_dados(dados: pd.DataFrame) -> pd.DataFrame`
Transforma os dados em um formato adequado para análise com Apriori.

- **Entrada**: DataFrame com dados.
- **Saída**: DataFrame transformado em formato de "cesta de compras", pronto para o algoritmo Apriori.

### 3. `aplicar_apriori(cesta: pd.DataFrame, suporte_minimo: float = 0.01) -> pd.DataFrame`
Aplica o algoritmo Apriori para encontrar conjuntos frequentes no conjunto de dados.

- **Entrada**: DataFrame de cestas e o suporte mínimo para os conjuntos frequentes.
- **Saída**: DataFrame com os conjuntos frequentes encontrados.

### 4. `gerar_regras(conjuntos_frequentes: pd.DataFrame, num_itemsets: int, metric: str = 'confidence', min_threshold: float = 0.5) -> pd.DataFrame`
Gera regras de associação a partir dos conjuntos frequentes.

- **Entrada**: DataFrame com conjuntos frequentes, número de itemsets, métrica de avaliação (confidence, lift, etc.), e o limiar mínimo.
- **Saída**: DataFrame com as regras geradas.

### 5. `gerar_grafico_comparacao_sexo(dados: pd.DataFrame)`
Gera um gráfico de barras para comparar a quantidade de formandos por sexo.

- **Entrada**: DataFrame com a coluna `SEXO` e `FORMADOS`.
- **Saída**: Gráfico de barras com a comparação de formandos por sexo.

### 6. `explorar_dados(dados: pd.DataFrame) -> None`
Realiza uma exploração preliminar dos dados, mostrando estatísticas descritivas e informações gerais.

- **Entrada**: DataFrame com dados.
- **Saída**: Visualizações gráficas e informações sobre os dados.

### 7. `limpar_dados(dados: pd.DataFrame) -> pd.DataFrame`
Remove duplicatas e preenche valores ausentes nos dados.

- **Entrada**: DataFrame com dados.
- **Saída**: DataFrame com dados limpos.

### 8. `transformar_dados(dados: pd.DataFrame) -> pd.DataFrame`
Transforma os dados para que possam ser usados em modelos de aprendizado de máquina, normalizando variáveis numéricas e codificando variáveis categóricas.

- **Entrada**: DataFrame com dados.
- **Saída**: DataFrame com dados transformados.

### 9. `reduzir_dimensao(dados: pd.DataFrame, n_componentes: int) -> pd.DataFrame`
Reduz a dimensionalidade dos dados utilizando PCA (Análise de Componentes Principais).

- **Entrada**: DataFrame com dados e o número de componentes principais.
- **Saída**: DataFrame com dados de dimensionalidade reduzida.

### 10. `treinar_modelo(dados: pd.DataFrame, alvo: str) -> None`
Treina um modelo de aprendizado de máquina (Random Forest) para prever o valor da coluna alvo e gera gráficos de avaliação.

- **Entrada**: DataFrame com dados transformados e o nome da coluna alvo.
- **Saída**: Relatório de classificação, matriz de confusão e gráficos de avaliação.

## Função Principal

A função principal (`main()`) permite ao usuário escolher entre duas abordagens:

1. **Árvore de Decisão**: Utiliza Random Forest para prever o valor de uma coluna alvo.
2. **Apriori**: Aplica o algoritmo Apriori para encontrar regras de associação entre itens em um conjunto de dados.

### Fluxo de Execução

1. O usuário escolhe o método desejado (Árvore de Decisão ou Apriori).
2. Os dados são carregados do arquivo.
3. Dependendo da escolha, o código executa a análise de associação ou o treinamento de um modelo de árvore de decisão.
4. São gerados gráficos para ajudar na visualização dos resultados, como gráficos de barras e pizza.

## Requisitos

Para rodar o código, você precisa das seguintes bibliotecas:

- `pandas`
- `matplotlib`
- `seaborn`
- `mlxtend`
- `sklearn`

Estas bibliotecas podem ser instaladas utilizando o comando `pip install pandas matplotlib seaborn mlxtend scikit-learn`.

## Conclusão

Este código oferece uma implementação abrangente para análise de dados utilizando tanto técnicas de aprendizado de máquina quanto de mineração de padrões. Ele gera insights visuais sobre os dados e pode ser facilmente adaptado para diferentes conjuntos de dados e problemas analíticos.
