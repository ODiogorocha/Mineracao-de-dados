import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix

# Ignorar alertas e permitir conexões SSL não verificadas
import warnings
import ssl
warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context

# Carregar o dataset
url_dataset = 'https://raw.githubusercontent.com/lopes-andre/datasets/main/credito.csv'
data = pd.read_csv(url_dataset)

# Exibição dos primeiros dados e resumo estatístico
print(f'Shape dos dados: {data.shape}\n')
print(f'Esta base de dados tem {data.shape[0]} linhas e {data.shape[1]} colunas.')
print(data.describe())

# Verificar as variáveis categóricas e suas contagens
colunas_cat = data.select_dtypes(include=['object']).columns.tolist()
for coluna in colunas_cat:
    print(f'### Coluna <{coluna}> ###')
    print(data[coluna].value_counts())
    print('-' * 40)

# Verificar tipos e valores nulos
print(data.info())
print('Colunas com dados nulos:')
print(data.isnull().sum()[data.isnull().sum() > 0])

# Corrigir erro de digitação na coluna 'motivo'
corrige_carro = {'carr0': 'carro'}
data.replace(corrige_carro, inplace=True)

# Verificar novamente as categorias da coluna 'motivo'
print(data['motivo'].value_counts())

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
    'telefone': {'nao': 1, 'sim': 2}
}

data.replace(conversao_variaveis, inplace=True)

# Exemplo de dados após conversão
print(data.sample(5))

# Realizar OneHotEncoding para as variáveis categóricas restantes
cols_cat = data.select_dtypes(include='object').columns.tolist()
cols_cat.remove('inadimplente')  # Remover a variável alvo
data = pd.get_dummies(data, columns=cols_cat, drop_first=True)

# Conversão da variável alvo 'inadimplente' para 0 e 1
conversao_alvo = {'inadimplente': {'nao': 0, 'sim': 1}}
data.replace(conversao_alvo, inplace=True)

# Imputação de valores nulos com a média
data = data.fillna(data.mean())

# Divisão dos dados em características (X) e alvo (y)
X = data.drop(['inadimplente'], axis=1)
y = data['inadimplente']

# Divisão entre treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)

# Verificando a proporção de classes nos dados de treino e teste
print('### Proporção de Classes em Treino ###')
print(f'Porcentagem de entradas Classe 0: {y_train.value_counts(normalize=True).values[0] * 100}%')
print(f'Porcentagem de entradas Classe 1: {y_train.value_counts(normalize=True).values[1] * 100}%\n')

print('### Proporção de Classes em Teste ###')
print(f'Porcentagem de entradas Classe 0: {y_test.value_counts(normalize=True).values[0] * 100}%')
print(f'Porcentagem de entradas Classe 1: {y_test.value_counts(normalize=True).values[1] * 100}%')

# Função para calcular as métricas de desempenho do modelo
def performance_modelo_classificacao(model, flag=True):
    score_list = []
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_test)

    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_test, y_test)

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
        print(f'\nRecall na base de Treino: {train_recall}')
        print(f'Recall na base de Teste: {val_recall}')
        print(f'\nPrecisão na base de Treino: {train_prec}')
        print(f'Precisão na base de Teste: {val_prec}')
        print(f'\nF1-Score na base de Treino: {train_f1}')
        print(f'F1-Score na base de Teste: {val_f1}')

    return score_list

# Função para plotar a matriz de confusão
def matriz_confusao(model, X, y_actual):
    y_predict = model.predict(X)
    cm = confusion_matrix(y_actual, y_predict, labels=[0, 1])
    df_cm = pd.DataFrame(cm, index=['Real - Não (0)', 'Real - Sim (1)'],
                         columns=['Previsto - Não (0)', 'Previsto - Sim (1)'])

    group_counts = [f'{value:.0f}' for value in cm.flatten()]
    group_percentages = [f'{value:.2f}%' for value in (cm.flatten() / np.sum(cm)) * 100]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=labels, fmt='')
    plt.xlabel('Classe Prevista', fontweight='bold')
    plt.ylabel('Classe Verdadeira', fontweight='bold')
    plt.show()

# Função para plotar as importâncias das variáveis
def importancias_variaveis(model):
    importances = model.feature_importances_
    indices = np.argsort(importances)
    feature_names = list(X.columns)

    plt.figure(figsize=(12, 12))
    plt.barh(y=range(len(indices)), width=importances[indices], color='violet', align='center')
    plt.title('Importância do Atributo', fontsize=16, fontweight='bold')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importância Relativa', fontweight='bold')
    plt.show()

# Treinamento do Modelo de Árvore de Decisão (sem podar)
arvore_d = DecisionTreeClassifier(random_state=1)
arvore_d.fit(X_train, y_train)
arvore_d_scores = performance_modelo_classificacao(arvore_d)

# Matrizes de Confusão
matriz_confusao(arvore_d, X_train, y_train)
matriz_confusao(arvore_d, X_test, y_test)

# Visualização da Árvore
plt.figure(figsize=(20, 30))
plot_tree(arvore_d, feature_names=X_train.columns, filled=True, fontsize=9, node_ids=True, class_names=True)

# Modelo de Árvore de Decisão com podas (max_depth=3)
arvore_d1 = DecisionTreeClassifier(random_state=1, max_depth=3)
arvore_d1.fit(X_train, y_train)
arvore_d1_scores = performance_modelo_classificacao(arvore_d1)

# Matrizes de Confusão
matriz_confusao(arvore_d1, X_train, y_train)
matriz_confusao(arvore_d1, X_test, y_test)

# Visualização da Árvore
plt.figure(figsize=(15, 10))
plot_tree(arvore_d1, feature_names=X_train.columns, filled=True, fontsize=9, node_ids=True, class_names=True)

# Grid Search para ajuste de hiperparâmetros
parameters = {
    'max_depth': np.arange(1, 10),
    'min_samples_leaf': [1, 2, 5, 7, 10, 15, 20],
    'max_leaf_nodes': [2, 3, 5, 10],
    'min_impurity_decrease': [0.001, 0.01, 0.1]
}

acc_scorer = metrics.make_scorer(metrics.recall_score)
grid_obj = GridSearchCV(DecisionTreeClassifier(random_state=1), parameters, scoring=acc_scorer, cv=5)
grid_obj.fit(X_train, y_train)

# Melhor modelo encontrado pelo GridSearch
arvore_d2 = grid_obj.best_estimator_
arvore_d2.fit(X_train, y_train)
arvore_d2_scores = performance_modelo_classificacao(arvore_d2)

# Matrizes de Confusão
matriz_confusao(arvore_d2, X_train, y_train)
matriz_confusao(arvore_d2, X_test, y_test)

# Comparação entre os modelos
modelos = ['Árvore de Decisão', 'Árvore de Decisão Podada', 'Árvore de Decisão Tunada']
colunas = ['Treino_Acuracia', 'Val_Acuracia', 'Treino_Recall', 'Val_Recall',
           'Treino_Precisao', 'Val_Precisao', 'Treino_F1', 'Val_F1']

modelos_scores = pd.DataFrame([arvore_d_scores, arvore_d1_scores, arvore_d2_scores],
                             columns=colunas, index=modelos).apply(lambda x: round(x, 2))

# Exibe a tabela com as métricas de todos os modelos
print(modelos_scores.T)