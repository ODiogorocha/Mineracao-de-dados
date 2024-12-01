import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import statsmodels.api as sm

# Dados de exemplo
x = np.array([201, 225, 305, 380, 560, 600, 685, 735, 510, 725, 450, 370, 150])
y = np.array([17, 20, 21, 23, 25, 24, 27, 27, 22, 30, 21, 15, 15])

# 1. Coeficientes beta do modelo de regressão estimado
X = sm.add_constant(x)  # Adiciona a constante para o intercepto
model = sm.OLS(y, X).fit()
print("Coeficientes beta (intercepto e inclinação):")
print(model.params)

# 2. Resumo do modelo de regressão estimado
print("\nResumo do modelo de regressão:")
print(model.summary())

# 3. Coeficiente de correlação de Pearson entre x e y
correlation = np.corrcoef(x, y)[0, 1]
print("\nCoeficiente de correlação de Pearson:", correlation)

# 4. Diagrama de dispersão com x no eixo horizontal e y no eixo vertical
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x, y=y)
plt.xlabel('Vendas (x)')
plt.ylabel('Lucro (y)')
plt.title('Diagrama de dispersão')
plt.show()

# 5. Diagrama de dispersão com x no eixo vertical e y no eixo horizontal
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y, y=x)
plt.xlabel('Lucro (y)')
plt.ylabel('Vendas (x)')
plt.title('Diagrama de dispersão invertido')
plt.show()

# 6. Teste de hipóteses para a média de x (mu), com variância desconhecida e H0: mu = 0
t_stat, p_value = stats.ttest_1samp(x, 0)
print("\nTeste de hipóteses para a média de x (H0: mu = 0):")
print("Estatística t:", t_stat)
print("p-valor:", p_value)

# 7. Teste de hipóteses para a média de x (mu), com variância desconhecida e H0: mu = 10
t_stat, p_value = stats.ttest_1samp(x, 10)
print("\nTeste de hipóteses para a média de x (H0: mu = 10):")
print("Estatística t:", t_stat)
print("p-valor:", p_value)
