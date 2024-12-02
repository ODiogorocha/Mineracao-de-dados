from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import matplotlib.pyplot as plt

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
    """
    Verifica o formato dos dados e transforma para o formato de cestas (basket) necessário para o Apriori.
    """
    print("Colunas disponíveis no dataset:", dados.columns.tolist())
    
    # Exibindo um exemplo dos dados para inspeção
    print("\nExemplo de dados:")
    print(dados.head())

    try:
        # Para criar cestas, vamos transformar as colunas em uma matriz de transações
        cesta = dados.apply(lambda x: x.astype(str))
        cesta_transformed = pd.get_dummies(cesta, columns=dados.columns)
        print("\nDados transformados em matriz de cestas:")
        print(cesta_transformed.head())
        return cesta_transformed
    except Exception as e:
        print(f"Erro ao transformar os dados para cestas: {e}")
        raise

def aplicar_apriori(cesta: pd.DataFrame, suporte_minimo: float = 0.01) -> pd.DataFrame:
    """
    Aplica o algoritmo Apriori para encontrar conjuntos frequentes.
    """
    try:
        conjuntos_frequentes = apriori(cesta, min_support=suporte_minimo, use_colnames=True)
        print("\nConjuntos frequentes encontrados:")
        print(conjuntos_frequentes)
        return conjuntos_frequentes
    except Exception as e:
        print(f"Erro ao aplicar o Apriori: {e}")
        raise

def gerar_regras(conjuntos_frequentes: pd.DataFrame, num_itemsets: int, metric: str = 'confidence', min_threshold: float = 0.5) -> pd.DataFrame:
    """
    Gera regras de associação a partir dos conjuntos frequentes.
    """
    try:
        # Use num_itemsets as an argument in the function call
        regras = association_rules(conjuntos_frequentes, metric=metric, min_threshold=min_threshold, num_itemsets=num_itemsets)
        print("\nRegras de associação encontradas:")
        print(regras)
        return regras
    except Exception as e:
        print(f"Erro ao gerar regras: {e}")
        raise

def gerar_grafico_comparacao_sexo(dados: pd.DataFrame):
    """
    Gera um gráfico de barras comparando o número de formandos do sexo masculino e feminino.
    """
    try:
        # Agrupar dados pelo sexo e somar os formandos
        comparacao_sexo = dados.groupby('SEXO')['FORMADOS'].sum()
        
        # Criar gráfico de barras
        comparacao_sexo.plot(kind='bar', title='Comparação de Formandos por Sexo')
        plt.xlabel('Sexo')
        plt.ylabel('Número de Formandos')
        plt.show()
    except Exception as e:
        print(f"Erro ao gerar o gráfico de comparação: {e}")
        raise

def main():
    caminho_dados = "arq/Ingressantes e Formandos/CT Ingressantes e formados por sexo.xls"

    print("Escolha o modelo de treinamento:")
    print(" (1) Árvore de Decisão")
    print(" (2) Apriori")
    try:
        Modelo = int(input("Digite o número do modelo desejado: "))
        
        if Modelo == 2:
            print("Iniciando Apriori...")
            try:
                # Carregar os dados
                dados = carregar_dados(caminho_dados)

                # Verificar e transformar os dados
                cesta = verificar_e_transformar_dados(dados)

                # Aplicar Apriori
                suporte_minimo = 0.02  # Ajuste conforme necessário
                conjuntos_frequentes = aplicar_apriori(cesta, suporte_minimo)

                # Gerar regras de associação
                num_itemsets = len(conjuntos_frequentes)
                regras = gerar_regras(conjuntos_frequentes, num_itemsets, metric='lift', min_threshold=1.0)

                # Gerar gráfico de comparação de sexo
                gerar_grafico_comparacao_sexo(dados)

            except Exception as e:
                print(f"Erro durante a execução do Apriori: {e}")

        else:
            print("Opção inválida! Por favor, escolha 1 ou 2.")
    except ValueError:
        print("Entrada inválida! Por favor, insira apenas números.")

if __name__ == "__main__":
    main()
