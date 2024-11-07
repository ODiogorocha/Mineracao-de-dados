import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

arquivo = pd.read_csv('Exercicios /GeyserUFSM.csv')

espera = arquivo[['espera']]  
erupcao = arquivo[['erupcao']]  

espera_treino, espera_teste, erupcao_treino, erupcao_teste = train_test_split(espera, erupcao, test_size=0.5, random_state=45)

modelo = LinearRegression()
modelo.fit(espera_treino, erupcao_treino)

erupcao_previsao = modelo.predict(espera_teste)

mse = mean_squared_error(erupcao_teste, erupcao_previsao)
print(f'Erro quadrático médio (MSE): {mse}')

espera_teste = espera_teste.values.ravel()
erupcao_teste = erupcao_teste.values.ravel()
erupcao_previsao = erupcao_previsao.flatten()

plt.scatter(espera_treino.values.ravel(), erupcao_treino.values.ravel(), color='blue', label='Dados de Treino')
plt.scatter(espera_teste, erupcao_teste, color='red', label='Dados de Teste')
plt.plot(espera_teste, erupcao_previsao, color='blue', linewidth=2, label='Previsão')
plt.xlabel('Tempo de Espera (min)')
plt.ylabel('Duração da Erupção (min)')
plt.legend()
plt.title("Previsão da Duração da Erupção do Gêiser")
plt.show()
