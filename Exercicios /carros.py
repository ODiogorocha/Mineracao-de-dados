import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Carregar o dataset mtcars
mtcars = sns.load_dataset('mpg').dropna()

# Modificar o dataset para as colunas desejadas
mtcars['am'] = mtcars['origin'].apply(lambda x: 1 if x == 'manual' else 0)  # Converte origin para manual (1) e automático (0)

# Converter 'mpg' de milhas por galão para km/L (1 milha/galão ≈ 0.425 km/L)
mtcars['km_per_l'] = mtcars['mpg'] * 0.425144
mtcars['wt_kg'] = mtcars['weight'] * 0.453592  # Converte peso de libras para kg

# Visualizar os dados
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wt_kg', y='km_per_l', hue='am', data=mtcars)
plt.xlabel('Peso (kg)')
plt.ylabel('Consumo (km/L)')
plt.title('Consumo vs Peso para Carros com Câmbio Manual e Automático')
plt.legend(title='Câmbio', labels=['Automático', 'Manual'])
plt.show()

# Preparar os dados para o modelo linear
X = mtcars[['wt_kg', 'am']]  # Variáveis preditoras: peso e tipo de câmbio
y = mtcars['km_per_l']  # Variável alvo: eficiência de combustível

# Adicionar uma constante para o modelo
X = sm.add_constant(X)

# Criar o modelo linear
modelo = sm.OLS(y, X).fit()

# Imprimir o resumo do modelo
print(modelo.summary())

# Prever a eficiência de combustível (km/L) para o conjunto de dados
mtcars['previsao_km_per_l'] = modelo.predict(X)

# Visualizar as previsões
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wt_kg', y='km_per_l', hue='am', data=mtcars, marker='o', label='Real')
sns.lineplot(x='wt_kg', y='previsao_km_per_l', data=mtcars, color='red', label='Previsão Linear')
plt.xlabel('Peso (kg)')
plt.ylabel('Consumo (km/L)')
plt.title('Previsão de Consumo vs Peso para Carros com Câmbio Manual e Automático')
plt.legend()
plt.show()
