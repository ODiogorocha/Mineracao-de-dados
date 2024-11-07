# Importação das bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    f1_score, accuracy_score, recall_score, precision_score,
    confusion_matrix, ConfusionMatrixDisplay, make_scorer
)
from typing import Optional
import warnings
import ssl

# Ignora alertas e configura o SSL
warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context

# Carrega o dataset
url_dataset = 'https://raw.githubusercontent.com/lopes-andre/datasets/main/credito.csv'
data = pd.read_csv(url_dataset)

# Visualização das primeiras linhas do dataset
print("Visualização das primeiras linhas do dataset:")
print(data.head(), "\n")
print(f"Shape dos dados: {data.shape}") 
print(f"Esta base de dados tem {data.shape[0]} linhas e {data.shape[1]} colunas.\n")
print("Descrição estatística dos dados numéricos:")
print(data.describe(), "\n")

# Verificação e correção de valores nas colunas categóricas
colunas_cat = data.select_dtypes(include=['object']).columns.tolist()
print("Contagem de valores únicos em colunas categóricas:")
for coluna in colunas_cat:
    print(f'### Coluna <{coluna}> ###')
    print(data[coluna].value_counts())
    print('-' * 40)

# Corrige erro de digitação
corrige_carro = {'carr0': 'carro'}
data.replace(corrige_carro, inplace=True)

# Conversão de variáveis categóricas ordinais
conversao_variaveis = {
    'saldo_corrente': {
        'desconhecido': -1, '< 0 DM': 1, '1 - 200 DM': 2, '> 200 DM': 3,
    },
    'historico_credito': {
        'critico': 1, 'ruim': 2, 'bom': 3, 'muito bom': 4, 'perfeito': 5
    },
    'saldo_poupanca': {
        'desconhecido': -1, '< 100 DM': 1, '100 - 500 DM': 2, '500 - 1000 DM': 3, '> 1000 DM': 4,
    },
    'tempo_empregado': {
        'desempregado': 1, '< 1 ano': 2, '1 - 4 anos': 3, '4 - 7 anos': 4, '> 7 anos': 5,
    },
    'telefone': {
        'nao': 1, 'sim': 2,
    }
}
data.replace(conversao_variaveis, inplace=True)

# Aplicação de OneHotEncoding nas variáveis categóricas restantes
cols_cat = data.select_dtypes(include='object').columns.tolist()
cols_cat.remove('inadimplente')
data = pd.get_dummies(data, columns=cols_cat, drop_first=True)

# Convertendo a variável alvo
data['inadimplente'] = data['inadimplente'].map({'nao': 0, 'sim': 1})

# Imputação de valores nulos
data.fillna(data.mean(), inplace=True)

# Divisão entre variáveis independentes (X) e dependente (y)
X = data.drop(['inadimplente'], axis=1)
y = data['inadimplente']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)

# Função para calcular as métricas de performance do modelo
def performance_modelo_classificacao(model: object, flag: Optional[bool] = True):
    score_list = []
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_test)
    train_acc = accuracy_score(y_train, pred_train)
    val_acc = accuracy_score(y_test, pred_val)
    train_recall = recall_score(y_train, pred_train)
    val_recall = recall_score(y_test, pred_val)
    train_prec = precision_score(y_train, pred_train)
    val_prec = precision_score(y_test, pred_val)
    train_f1 = f1_score(y_train, pred_train)
    val_f1 = f1_score(y_test, pred_val)
    score_list.extend((train_acc, val_acc, train_recall, val_recall, train_prec, val_prec, train_f1, val_f1))
    
    if flag:
        print(f'Acurácia na base de Treino: {train_acc}')
        print(f'Acurácia na base de Teste: {val_acc}')
        print(f'Recall na base de Treino: {train_recall}')
        print(f'Recall na base de Teste: {val_recall}')
        print(f'Precisão na base de Treino: {train_prec}')
        print(f'Precisão na base de Teste: {val_prec}')
        print(f'F1-Score na base de Treino: {train_f1}')
        print(f'F1-Score na base de Teste: {val_f1}')
    
    return score_list

# Função para exibir a matriz de confusão
def matriz_confusao(model, X, y):
    pred = model.predict(X)
    cm = confusion_matrix(y, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.show()

# Instancia e treina o modelo inicial
arvore_d = DecisionTreeClassifier(random_state=1)
arvore_d.fit(X_train, y_train)
arvore_d_scores = performance_modelo_classificacao(arvore_d)

# Ajuste com Grid Search
parameters = {
    'max_depth': np.arange(1, 10),
    'min_samples_leaf': [1, 2, 5, 7, 10, 15, 20],
    'max_leaf_nodes': [2, 3, 5, 10],
    'min_impurity_decrease': [0.001, 0.01, 0.1]
}
grid_obj = GridSearchCV(arvore_d, parameters, scoring=make_scorer(recall_score), cv=5)
grid_obj.fit(X_train, y_train)

# Melhor modelo ajustado
arvore_d2 = grid_obj.best_estimator_
arvore_d2.fit(X_train, y_train)
arvore_d2_scores = performance_modelo_classificacao(arvore_d2)

# Obtenção dos valores de ccp_alpha para poda
path = arvore_d2.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Testa os valores de ccp_alpha e armazena as métricas
modelos_podados = []
scores_podados = []
for ccp_alpha in ccp_alphas:
    arvore_podada = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
    arvore_podada.fit(X_train, y_train)
    modelos_podados.append(arvore_podada)
    scores_podados.append(performance_modelo_classificacao(arvore_podada, flag=False))

# Escolhe o modelo com melhor F1-Score na validação
melhor_score_val = max(scores_podados, key=lambda x: x[7])  # 7 é o índice do F1-Score de validação
melhor_alpha = ccp_alphas[scores_podados.index(melhor_score_val)]
arvore_podada_final = DecisionTreeClassifier(random_state=1, ccp_alpha=melhor_alpha)
arvore_podada_final.fit(X_train, y_train)

# Exibe a árvore podada
print(f"Melhor valor de ccp_alpha para poda: {melhor_alpha}")
plt.figure(figsize=(20, 30))
plot_tree(arvore_podada_final, feature_names=X.columns, filled=True, fontsize=9, node_ids=True, class_names=True)
plt.show()

# Avaliação do modelo podado final
print("\nDesempenho do modelo podado:")
performance_modelo_classificacao(arvore_podada_final)
matriz_confusao(arvore_podada_final, X_train, y_train)
matriz_confusao(arvore_podada_final, X_test, y_test)
