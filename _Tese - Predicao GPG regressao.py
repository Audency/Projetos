#!/usr/bin/env python
# coding: utf-8

# # Prediction of Gestational Weight Gain for Pregnancy: A Machine Learning-regression.
# 
# #### Audencio Victor 
# 
# ##### Dataset: Coorte-Araraquara
# 
# ##### Objetivo da Análise: Identificação precoce de mulheres com maior risco de ganho de peso excessivo durante a gestação pode permitir intervenções precoces para prevenir complicações e promover um ganho de peso adequado
# 
# ##### O Conjunto de dados possui 1455 observações, com as as seguintes variáveis listadas:

# ### O Conjunto de dados possui 1455 observações, com as as seguintes variáveis listadas:
#  1. a_idade - Idade materna,
#  2. a_estat1 - Altura, 
#  3. a_pesopre - peso pregestational, 
#  4. a_imcpg - IMC-pregest,  
#  5. a_cor -Raca,((1) branca / (2) preta / (3) amarela / (4) indígena / (5) parda)  
#  6. a_vigord -Actividade fisica 
#  7. a_fumog - Fumar
#  8. a_igadums -Idade gestacional em semanas,
#  9. a_agdm -Diabetes 
#  10. a_aghas -Hipertensao
#  11. a_npari- Numero de gravidezes anterior 
#  12. a_escola - Escolaridade  da gestante (em anos de estudos)
#  13. a_civil - Estado civil  (1) casada  e solteira (com companheiro) / (2) solteira (sem companheiro) /separada/viúva
#  14. a_hba1c - hemoglobina glicada, 
#  15. a_hdl- HDL
#  16. a_ldl-LDL
#  17. a_ct - colestreol
#  18. gpg - Ganho de peso gestacional 

# ### Roteiro:
# 1. Importação dos Pacotes
# 2. Chamada do Conjunto de Dados
# 3. Transformação das Variáveis
# 4. Separação em treino e teste - Split aplicado de 80%
# 5. Retirada de variáveis indesejadas
# 6. Correlação/Teste de dependência *_(novo)_*
# 7. Gráficos de frequências para X_train
# 8. One Hot Encoding
# 9. Treinamento do Primeiro Modelo (Entrega 2)
# 10. Obtenção das métricas com base no conjunto de teste (Entrega 2)
# 11. Treinando e comparando múltiplos algoritmos (Entrega 3)
# 13. Feature Selection - Boruta (Entrega 3)
# 14. Modelo de Regressão 
# 15. Boruta
# 16. Próximos passos

# ####  Importação dos Pacotes

# In[3]:


get_ipython().system('pip install pycaret')
get_ipython().system('pip install lazypredict')
get_ipython().system('pip install dfply')
get_ipython().system('pip install pyforest')
get_ipython().system('pip install boruta')
get_ipython().system('pip install dfply ')
get_ipython().system('pip install scikit-plot')
get_ipython().system('pip install graphviz')
get_ipython().system('pip install dtreeviz')
get_ipython().system("pip install dfply # utilizamos para ativar o 'pipe' no python, denotado por >>, além de algumas outras funções para manipulação dos dados")
get_ipython().system('pip install yellowbrick # visualização de gráficos para machine learning')


# ##### Importação de bibliotecas

# In[4]:


import pandas as pd # para processamento de bancos de dados
import numpy as np # para processamento numérico de bancos de dados
from dfply import *  # para importar os comandos da biblioteca dfply
import matplotlib.pyplot as plt # para geração de gráficos
from matplotlib import rc  # configurações adicionais para os gráficos a serem gerados

# informamos ao Python que estamos usando um notebook e que os gráficos devem ser exibidos nele
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #alternativa para a matplotlib para geração de gráficos

# definimos o estilo dos gráficos
# mais estilos em https://matplotlib.org/3.1.1/gallery/#style-sheets
plt.style.use("fivethirtyeight") 
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina' # formato das imagens")
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':10}) #fonte utilizada
rc('mathtext',**{'default':'regular'})

import warnings   # ignorando os warnings emitidos pelo Python
warnings.filterwarnings("ignore")

import operator  # para ordenação do zip

np.random.seed(45)  # semente de aleatoriedade


# #### Chamada do Conjunto de Dados

# In[1]:


df = pd.read_csv("dadosML1")
df.head()


# In[5]:


df.tail
df.columns  # para ver os nomes das  variveis 
list(df.columns.values.tolist()) # listar todas as variaveis 


# In[2]:


df.info() # tomando informações sobre o tipo das variáveis parametrizadas na chamada


# In[319]:


# estatísticas básicas do conjuntos de dados
df.describe().T


# 
# ###### Para este estudo teremos ganho de peso numerico 

# In[249]:


# variável de interesse/desfecho 
target= df >> select(X.gpg)


# In[320]:


print(df)


# In[321]:


#Apagando variaveis do dataset 
df.drop(["cat_imc"  , "categoria", "d_PesoParto", "a_pesopre"], axis=1, inplace=True)


# In[322]:


df.head()


# In[323]:


# Para as variáveis categóricas iremos criar dummies/ One hotencode
df= pd.get_dummies(df, columns=['a_idade','a_fumog', 'a_agdm','a_aghas', 'a_civil', 'a_cor'  ])


# In[324]:


df.tail()


# In[325]:


df.isna()
df.isna().sum()


# In[326]:


##imputar dados pela media dsa variavel (Separar entre treino e teste)
df.fillna(df.mean(), inplace=True)


# ### Separação entre treino e teste

# In[350]:


variaveis_preditoras = df.iloc[:, df.columns != 'gpg']
classe = df.iloc[:, df.columns == 'gpg']
X_train, X_test, y_train, y_test = train_test_split(variaveis_preditoras, 
                                                    classe,
                                                    train_size = 0.60,
                                                    random_state = 45)


# In[351]:


X_train.shape


# In[352]:


X_test.shape


# In[353]:


X_test.columns


# In[354]:


X_train.columns


# In[355]:


# Standarscaler com passthrough tem um problema de ordenação das colunas. Quando aplicamos, ele fornce o resultado com as colunas padronizadas em primeiro, seguidas das demais colunas.
# Para resolver este problema, iremos ordenar as nossas colunas alocando as contínuas nas primeiras posições 

X_train = X_train.loc[:,[ 'a_estat1', 'a_pesoat', 'a_rendpcr', 'a_fmp', 'a_vigord', 'a_moderd',
       'a_igadums', 'a_escola', 'a_hba1c', 'a_hdl', 'a_ldl', 'a_ct', 'imc',
       'a_idade_1', 'a_idade_2', 'a_idade_3', 'a_fumog_0', 'a_fumog_1',
       'a_agdm_0', 'a_agdm_1', 'a_aghas_0', 'a_aghas_1', 'a_civil_1',
       'a_civil_2', 'a_civil_3', 'a_civil_4', 'a_cor_1', 'a_cor_2', 'a_cor_4',
       'a_cor_5']]

X_test = X_test.loc[:,[ 'a_estat1', 'a_pesoat', 'a_rendpcr', 'a_fmp', 'a_vigord', 'a_moderd',
       'a_igadums', 'a_escola', 'a_hba1c', 'a_hdl', 'a_ldl', 'a_ct', 'imc',
       'a_idade_1', 'a_idade_2', 'a_idade_3', 'a_fumog_0', 'a_fumog_1',
       'a_agdm_0', 'a_agdm_1', 'a_aghas_0', 'a_aghas_1', 'a_civil_1',
       'a_civil_2', 'a_civil_3', 'a_civil_4', 'a_cor_1', 'a_cor_2', 'a_cor_4',
       'a_cor_5']]

X_train_columns = X_train.columns
X_test_columns = X_test.columns


# #### Padronizacao das variveis Continuas

# In[356]:


from sklearn.compose import ColumnTransformer

### variáveis contínuas que serão padronizadas
continuous_cols = ['a_estat1', 'a_pesoat', 'a_rendpcr', 'a_fmp', 'a_vigord', 'a_moderd',
       'a_igadums', 'a_escola', 'a_hba1c', 'a_hdl', 'a_ldl', 'a_ct', 'imc']

def setScaler():
  ct = ColumnTransformer([
        ('scaler', StandardScaler(), continuous_cols)
    ], remainder='passthrough' # utilizamos para manter as colunas em que não aplicamos o scaler
  )
  return ct
  
scaler = setScaler()


# In[357]:


scaler.fit(X_train)


# In[358]:


X_train = scaler.transform(X_train)


# In[359]:


X_test = scaler.transform(X_test)


# In[360]:


# para evitarmos a exibição dos dados em notacao científica
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# In[361]:


X_train = pd.DataFrame(X_train, columns=X_train_columns)
X_test = pd.DataFrame(X_test, columns=X_test_columns)


# In[362]:


X_train.head()


# In[363]:


X_test.describe()


# In[343]:


get_ipython().system('pip install pycaret')
from pycaret.regression import *

# configurar experimento do PyCaret
clf = setup(df, target='gpg')

# comparar modelos
best_model = compare_models()


# ### Função auxiliar RunModel (Rodando o modelo )

# In[1025]:


import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def runModelRegressor(model, X_train, y_train, X_test, y_test, plot_residuals=True, title=""):
    """Função auxiliar para execução de modelos de regressão.
    
    Parâmetros:
    
    - model: modelo de regressão a ser executado
    - X_train: base de treinamento das variáveis preditoras
    - y_train: base de treinamento da variável de resposta
    - X_test: base de teste das variáveis preditoras
    - y_test: base de teste da variável de resposta
    - plot_residuals (default: True): define se será exibido o gráfico de resíduos
    - title: define o título a ser exibido nos gráficos
    """

    # Treina o modelo e faz as previsões
    clf = model
    name = title
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Calcula as métricas de avaliação
    print("%s:" % name)
    print("\tMAE: %1.3f" % mean_absolute_error(y_test, y_pred))
    print("\tMSE: %1.3f" % mean_squared_error(y_test, y_pred))
    print("\tR2: %1.3f\n" % r2_score(y_test, y_pred))

    # Plota o gráfico de resíduos, se desejado
    if plot_residuals:
        plt.scatter(y_pred, y_test - y_pred, c="steelblue", edgecolor="white", s=70)
        plt.xlabel("Previsões")
        plt.ylabel("Resíduos")
        plt.title(title)
        plt.show()


# In[1026]:


# Função de best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# #### Testando múltiplos algoritmos

# In[209]:


import pyforest
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


# In[366]:


reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Print the results
print(models)


# In[364]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression

# Criando um modelo de regressão Elastic Net com parâmetros de regularização alpha e l1_ratio
regr = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Treinando o modelo com os dados de treinamento
regr.fit(X_train, y_train)

# Avaliando o desempenho do modelo com os dados de teste
score = regr.score(X_test, y_test)
print(score)


# In[365]:



from pycaret.regression import *

# configurar experimento do PyCaret
clf = setup(df, target='gpg')

# comparar modelos
best_model = compare_models()


# In[348]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# y_test contém os valores verdadeiros e y_pred_test contém as previsões do modelo
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

plt.figure(figsize = (10,10))
plt.scatter(y_test, y_pred_test , c = 'black')
#p1 = max(np.concatenate((y_test, y_pred_test, np.array(y_test))), axis=None)
#p2 = min(np.concatenate((y_test, y_pred_test, np.array(y_test))), axis=None)
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
#plt.plot([p1, p2], [p1, p2], 'r--')
plt.axis('equal')
plt.title('True vs Predicted (Test set) - MSE: {:.2f}, R2: {:.2f}'.format(mse, r2), fontsize=20)
plt.show()


# In[ ]:


xgboost_model =  create_model('xgboost')
interpret_model(xgboost_model)


# ### Execução dos algoritmos de machine learning

# ### 12) Construção dos modelos
# 
# Obs: foram selecionados os seguintes modelos:
# * LGBMClassifer: 1o melhor acuracia  e baixo tempo de construção
# * XGBClassifier: 2 melhor AUC e baixo tempo de construção
# * Regresão Logística: 3o melhor AUC e baixo tempo de construção

# #### LGBMClassifier 

# In[1033]:


import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score

model = KernelRidge(kernel='linear')

#treinando o modelo
model.fit(X_train, y_train)

#fazendo as previsões
y_pred = model.predict(X_test)

#avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")


# In[662]:


from sklearn.metrics import roc_curve, auc
# calculando a curva AUROC
y_prob = model.predict_proba(X_test)
fpr = {}
tpr = {}
roc_auc = {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# plotando a curva AUROC
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.plot(fpr[0], tpr[0], color='blue', lw=2, label='Classe 0 (Abaixo)')
plt.plot(fpr[1], tpr[1], color='red', lw=2, label='Classe 1 (Dentro)')
plt.plot(fpr[2], tpr[2], color='green', lw=2, label='Classe 2 (Acima)')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo (FPR)')
plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
plt.title('LGBMClassifie')
plt.legend(loc="lower right")

# adicionando os valores de AUROC no gráfico
plt.text(0.5, 0.3, "AUROC Classe 0 (Abaixo) = {:.3f}".format(roc_auc[0]), ha='center', va='center', size=12, color='blue')
plt.text(0.3, 0.5, "AUROC Classe 1 (Dentro) = {:.3f}".format(roc_auc[1]), ha='center', va='center', size=12, color='red')
plt.text(0.6, 0.7, "AUROC Classe 2 (Acima) = {:.3f}".format(roc_auc[2]), ha='center', va='center', size=12, color='green')

plt.show()


# #### XGBOOST

# In[664]:


import xgboost as xgb
model = xgb.XGBClassifier(objective='multi:softmax', num_class=3)

# treinando o modelo
model.fit(X_train, y_train)

# fazendo as previsões
y_pred = model.predict(X_test)

# avaliando o modelo
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[665]:


from sklearn.metrics import roc_curve, auc
# calculando a curva AUROC
y_prob = model.predict_proba(X_test)
fpr = {}
tpr = {}
roc_auc = {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# plotando a curva AUROC
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.plot(fpr[0], tpr[0], color='blue', lw=2, label='Classe 0 (Abaixo)')
plt.plot(fpr[1], tpr[1], color='red', lw=2, label='Classe 1 (Dentro)')
plt.plot(fpr[2], tpr[2], color='green', lw=2, label='Classe 2 (Acima)')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo (FPR)')
plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
plt.title('XGBOOST')
plt.legend(loc="lower right")

# adicionando os valores de AUROC no gráfico
plt.text(0.5, 0.3, "AUROC Classe 0 (Abaixo) = {:.3f}".format(roc_auc[0]), ha='center', va='center', size=12, color='blue')
plt.text(0.3, 0.5, "AUROC Classe 1 (Dentro) = {:.3f}".format(roc_auc[1]), ha='center', va='center', size=12, color='red')
plt.text(0.6, 0.7, "AUROC Classe 2 (Acima) = {:.3f}".format(roc_auc[2]), ha='center', va='center', size=12, color='green')

plt.show()


# #### LogisticRegression

# In[666]:


from sklearn.linear_model import LogisticRegression

# Criando o modelo
model = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42)

# Treinando o modelo
model.fit(X_train, y_train)

# avaliando o modelo
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[667]:


# Calculando a curva AUROC
y_prob = model.predict_proba(X_test)
fpr = {}
tpr = {}
roc_auc = {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotando a curva AUROC
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.plot(fpr[0], tpr[0], color='blue', lw=2, label='Classe 0 (Abaixo)')
plt.plot(fpr[1], tpr[1], color='red', lw=2, label='Classe 1 (Dentro)')
plt.plot(fpr[2], tpr[2], color='green', lw=2, label='Classe 2 (Acima)')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo (FPR)')
plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
plt.title('LogisticRegression')
plt.legend(loc="lower right")


# adicionando os valores de AUROC no gráfico
plt.text(0.5, 0.3, "AUROC Classe 0 (Abaixo) = {:.3f}".format(roc_auc[0]), ha='center', va='center', size=12, color='blue')
plt.text(0.3, 0.5, "AUROC Classe 1 (Dentro) = {:.3f}".format(roc_auc[1]), ha='center', va='center', size=12, color='red')
plt.text(0.6, 0.7, "AUROC Classe 2 (Acima) = {:.3f}".format(roc_auc[2]), ha='center', va='center', size=12, color='green')

plt.show()


# ####  Seleção de Variáveis - BORUTA

# In[668]:


get_ipython().system('pip install boruta')


# In[669]:


from boruta import BorutaPy
from lightgbm import LGBMClassifier
import numpy as np

# criando estimador para o Boruta
lgbm = LGBMClassifier(
   n_jobs = -1,
   max_depth = 5
)

boruta = BorutaPy(
   estimator = lgbm, 
   n_estimators = 'auto',
   max_iter = 100 # número de tentativas a serem realizadas
)

# parametrizando para o conjunto de treino
boruta.fit(np.array(X_train), np.array(y_train))

# resultados
green_area = X_train.columns[boruta.support_].to_list()
blue_area = X_train.columns[boruta.support_weak_].to_list()
print('features na área verde:', green_area)
print('features na área azul:', blue_area)


# #### Retreinamento do modelos apos Boruta 

# In[670]:


get_ipython().system('pip install boruta')


# In[671]:


from boruta import BorutaPy
from lightgbm import LGBMClassifier
import numpy as np

# creating the estimator for Boruta
lgbm = LGBMClassifier(
   n_jobs = -1, 
   max_depth = 5
)

boruta = BorutaPy(
   estimator = lgbm, 
   n_estimators = 'auto',
   max_iter = 100 # number of trials to perform
)

# fitting to the training set
boruta.fit(np.array(X_train), np.array(y_train))

# results
green_area = X_train.columns[boruta.support_].to_list()
blue_area = X_train.columns[boruta.support_weak_].to_list()
print('features in the green area:', green_area)
print('features in the blue area:', blue_area)


# In[672]:


# Modelo com Boruta - Selecionado as variáveis preditoras
X_train_boruta = X_train[['a_pesoat', 'imc']]

X_test_boruta = X_test[['a_pesoat', 'imc']]

print('Shape sem boruta', X_train.shape, X_test.shape)
print('Shape com boruta', X_train_boruta.shape, X_test_boruta.shape)


# #### LGBM com Boruta

# In[673]:


#LGBM
clf_lgbm_boruta = LGBMClassifier()
clf_lgbm_boruta.fit(X_train_boruta, y_train)

y_pred_lgbm_boruta = clf_lgbm_boruta.predict(X_test_boruta)
prob_pos_lgbm_boruta = clf_lgbm_boruta.predict_proba(X_test_boruta)[:,1]

print('LGBM Model')
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_lgbm_boruta)))


print(classification_report(y_test, y_pred_lgbm_boruta))


# In[582]:


# Calculando a curva AUROC
y_prob = clf_lgbm_boruta.predict_proba(X_test_boruta)
fpr = {}
tpr = {}
roc_auc = {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotando a curva AUROC
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.plot(fpr[0], tpr[0], color='blue', lw=2, label='Classe 0 (Abaixo)')
plt.plot(fpr[1], tpr[1], color='red', lw=2, label='Classe 1 (Dentro)')
plt.plot(fpr[2], tpr[2], color='green', lw=2, label='Classe 2 (Acima)')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo (FPR)')
plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
plt.title('lgbm_boruta')
plt.legend(loc="lower right")


# adicionando os valores de AUROC no gráfico
plt.text(0.5, 0.3, "AUROC Classe 0 (Abaixo) = {:.3f}".format(roc_auc[0]), ha='center', va='center', size=12, color='blue')
plt.text(0.3, 0.5, "AUROC Classe 1 (Dentro) = {:.3f}".format(roc_auc[1]), ha='center', va='center', size=12, color='red')
plt.text(0.6, 0.7, "AUROC Classe 2 (Acima) = {:.3f}".format(roc_auc[2]), ha='center', va='center', size=12, color='green')

plt.show()


# ####  Xgboost com Boruta

# In[674]:


# xgboost
clf_xgboost_boruta = xgb.XGBClassifier()
clf_xgboost_boruta.fit(X_train_boruta, y_train)

y_pred_xgboost_boruta = clf_xgboost_boruta.predict(X_test_boruta)
prob_pos_xgboost_boruta = clf_xgboost_boruta.predict_proba(X_test_boruta)[:,1]

print('XGBoost Model')
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_xgboost_boruta)))


print(classification_report(y_test, y_pred_xgboost_boruta))


# In[675]:


# Calculando a curva AUROC
y_prob = clf_xgboost_boruta.predict_proba(X_test_boruta)
fpr = {}
tpr = {}
roc_auc = {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotando a curva AUROC
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.plot(fpr[0], tpr[0], color='blue', lw=2, label='Classe 0 (Abaixo)')
plt.plot(fpr[1], tpr[1], color='red', lw=2, label='Classe 1 (Dentro)')
plt.plot(fpr[2], tpr[2], color='green', lw=2, label='Classe 2 (Acima)')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo (FPR)')
plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
plt.title('Xgboost_boruta')
plt.legend(loc="lower right")


# adicionando os valores de AUROC no gráfico
plt.text(0.5, 0.3, "AUROC Classe 0 (Abaixo) = {:.3f}".format(roc_auc[0]), ha='center', va='center', size=12, color='blue')
plt.text(0.3, 0.5, "AUROC Classe 1 (Dentro) = {:.3f}".format(roc_auc[1]), ha='center', va='center', size=12, color='red')
plt.text(0.6, 0.7, "AUROC Classe 2 (Acima) = {:.3f}".format(roc_auc[2]), ha='center', va='center', size=12, color='green')

plt.show()


# ####  Logistic regression com Boruta

# In[676]:


# logistic regression
clf_lr_boruta = LogisticRegression()
clf_lr_boruta.fit(X_train_boruta, y_train)

y_pred_lr_boruta = clf_lr_boruta.predict(X_test_boruta)
prob_pos_lr_boruta = clf_lr_boruta.predict_proba(X_test_boruta)[:,1]

print('Logistic Regression Model')
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_lr_boruta)))


print(classification_report(y_test, y_pred_lr_boruta))


# In[433]:


# Calculando a curva AUROC
y_prob = clf_lr_boruta.predict_proba(X_test_boruta)
fpr = {}
tpr = {}
roc_auc = {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotando a curva AUROC
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.plot(fpr[0], tpr[0], color='blue', lw=2, label='Classe 0 (Abaixo)')
plt.plot(fpr[1], tpr[1], color='red', lw=2, label='Classe 1 (Dentro)')
plt.plot(fpr[2], tpr[2], color='green', lw=2, label='Classe 2 (Acima)')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo (FPR)')
plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
plt.title('logistic regression_boruta')
plt.legend(loc="lower right")


# adicionando os valores de AUROC no gráfico
plt.text(0.5, 0.3, "AUROC Classe 0 (Abaixo) = {:.3f}".format(roc_auc[0]), ha='center', va='center', size=12, color='blue')
plt.text(0.3, 0.5, "AUROC Classe 1 (Dentro) = {:.3f}".format(roc_auc[1]), ha='center', va='center', size=12, color='red')
plt.text(0.6, 0.7, "AUROC Classe 2 (Acima) = {:.3f}".format(roc_auc[2]), ha='center', va='center', size=12, color='green')

plt.show()


# #### Importancia das variaveis pelo Shapley

# In[677]:


get_ipython().system('pip install shap')
import shap


# In[678]:


# resultados no teste
shap_values_test = shap.TreeExplainer(clf_xgboost_boruta).shap_values(X_test_boruta)
shap.summary_plot(shap_values_test, X_test_boruta, plot_type="bar")


# In[ ]:





# In[ ]:




