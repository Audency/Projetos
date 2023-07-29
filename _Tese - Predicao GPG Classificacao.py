#!/usr/bin/env python
# coding: utf-8

# # Prediction of Gestational Weight Gain for Pregnancy: A Machine Learning-Based Multiclass Classification Approach.
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
# 12. Desenvolvendo algoritmo selecionado (Entrega 3)
#     * Logistic Regression
#     * LightGBM
#     * Adaboost
# 13. Feature Selection - Boruta (Entrega 3)
# 14. Modelo de Regressão Logística com Boruta (Entrega 3)
# 15. LGBM com Boruta (Entrega 3)
# 16. Próximos passos

# ####  Importação dos Pacotes

# In[1]:


get_ipython().system('pip install catboost')
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

# In[2]:


import catboost
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

# In[3]:


df = pd.read_csv("dadosML123")
df.head()


# In[4]:


df.tail
df.columns  # para ver os nomes das  variveis 
list(df.columns.values.tolist()) # listar todas as variaveis 


# In[5]:


df.info() # tomando informações sobre o tipo das variáveis parametrizadas na chamada


# #### Transformação das Variáveis
# O IOM (Institute of Medicine) publicou em 2009 e atualizou em 2019 suas recomendações para o ganho de peso durante a gestação com base no índice de massa corporal (IMC) pré-gestacional da mulher. Essas recomendações são baseadas em evidências e foram desenvolvidas para promover a saúde materna e fetal, reduzindo riscos à saúde associados ao ganho excessivo ou insuficiente de peso durante a gestação.
# 
# A classificação do ganho de peso gestacional segundo o IOM 2019, baseada no IMC pré-gestacional da mulher, é a seguinte:
# 
# IMC < 18,5 (baixo peso):
# Ganho de peso recomendado: 12,5 a 18 kg.
# IMC entre 18,5 e 24,9 (peso normal):
# 
# Ganho de peso recomendado: 11,5 a 16 kg.
# IMC entre 25 e 29,9 (sobrepeso):
# 
# Ganho de peso recomendado: 7 a 11,5 kg.
# IMC entre 30 e 34,9 (obesidade grau 1):
# 
# Ganho de peso recomendado: 5 a 9 kg.
# IMC entre 35 e 39,9 (obesidade grau 2):
# 
# Ganho de peso recomendado: 4 a 7 kg.
# IMC ≥ 40 (obesidade grau 3):
# 
# Ganho de peso recomendado: 3 a 6 kg.
# É importante lembrar que essas recomendações são gerais e podem ser ajustadas de acordo com a situação individual de cada mulher, devendo sempre ser acompanhadas pelo médico obstetra.
# ###### Para este estudo teremoos  3 categorias: Abaixo, acima e dentro das recoendacoes.

# In[6]:


# Instantiate a LabelEncoder object
le = LabelEncoder()


# In[7]:


# Trans formando a variavel categoria (GPG) em target
df['target'] = le.fit_transform(df['categoria'])
df['target']


# In[8]:


print(df)


# In[11]:


#Apagando variaveis do dataset 
df.drop(["gpg" , "categoria", "d_PesoParto",'gpg','categoria1',"a_agdm",'b_agdm', "a_hba1c",'a_insul','a_homa','f_FMP','imc'], axis=1, inplace=True)
 


# In[12]:


#Apagando variaveis do dataset 
df.drop(["gpg", "categoria","categoria1","cat_imc", "cat_imc", "a_cor", "a_civil", "a_alcool", "a_vigorh", "a_moderh", "a_pcr",
         "a_homa", "a_hba1c", "a_npcomo", "a_insul", "a_tg", "a_ct", "a_ldl", "b_agdm", "c_agdm", 
         "a_aghas", "a_agui", "a_agsif", "a_agceva", "a_fumog", "f_FMP", "f_FFMp", "d_PesoParto"], axis=1, inplace=True)


# In[13]:


df.head()


# In[14]:


# Para as variáveis categóricas iremos criar dummies/ One hotencode
df= pd.get_dummies(df, columns=["a_idade",'a_fumog', 'a_agdm','a_aghas',, "a_escola", 'a_civil', 'a_cor'])


# In[15]:


df.tail()


# In[9]:


df.isna()
df.isna().sum()


# In[13]:


##imputar dados pela media dsa variavel (Separar entre treino e teste)
df.fillna(df.mean(), inplace=True)


# #### Correlação/Teste de dependência

# In[14]:


from sklearn.feature_selection import chi2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Separação entre treino e teste

# In[15]:


variaveis_preditoras = df.iloc[:, df.columns != 'target']
classe = df.iloc[:, df.columns == 'target']
X_train, X_test, y_train, y_test = train_test_split(variaveis_preditoras, 
                                                    classe,
                                                    train_size = 0.80,
                                                    random_state = 45)


# In[16]:


X_train.shape


# In[17]:


X_test.shape


# In[18]:


X_test.columns


# In[19]:


X_train.columns


# In[20]:


# Standarscaler com passthrough tem um problema de ordenação das colunas. Quando aplicamos, ele fornce o resultado com as colunas padronizadas em primeiro, seguidas das demais colunas.
# Para resolver este problema, iremos ordenar as nossas colunas alocando as contínuas nas primeiras posições 

X_train = X_train.loc[:,['a_rendpcr', 'a_alcool', 'a_ngesta', 'a_estat1', 'a_circbracm', 'a_fmp',
       'a_vigord', 'a_moderd', 'd_IGUSGrn', 'a_pcr',
       'a_agdm_Sim', 'a_aghas_Não', 'a_aghas_Sim', 'a_escola_5-11',
       'a_escola_≤4', 'a_escola_≥12', 'a_civil_1', 'a_civil_2',
       'a_cor_Nbranco', 'a_cor_branco']]

X_test = X_test.loc[:,['a_rendpcr', 'a_alcool', 'a_ngesta', 'a_estat1', 'a_circbracm', 'a_fmp',
       'a_vigord', 'a_moderd', 'd_IGUSGrn', 'a_pcr',
       'a_agdm_Sim', 'a_aghas_Não', 'a_aghas_Sim', 'a_escola_5-11',
       'a_escola_≤4', 'a_escola_≥12', 'a_civil_1', 'a_civil_2',
       'a_cor_Nbranco', 'a_cor_branco']]

X_train_columns = X_train.columns
X_test_columns = X_test.columns


# #### Padronizacao das variveis Continuas

# In[31]:


from sklearn.compose import ColumnTransformer

### variáveis contínuas que serão padronizadas
continuous_cols = ['a_estat1', 'a_pesoat', 'a_igadums', 'imc']

def setScaler():
  ct = ColumnTransformer([
        ('scaler', StandardScaler(), continuous_cols)
    ], remainder='passthrough' # utilizamos para manter as colunas em que não aplicamos o scaler
  )
  return ct
  
scaler = setScaler()


# In[25]:


scaler.fit(X_train)


# In[26]:


X_train = scaler.transform(X_train)


# In[32]:


X_test = scaler.transform(X_test)


# In[721]:


# para evitarmos a exibição dos dados em notacao científica
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# In[722]:


X_train = pd.DataFrame(X_train, columns=X_train_columns)
X_test = pd.DataFrame(X_test, columns=X_test_columns)


# In[723]:


X_train.head()


# In[724]:


X_test.describe()


# ### Função auxiliar RunModel (Rodando o modelo )

# In[726]:


# Criando uma função para obtenção dos principais indicadores de performance do modelo
def runModel(model, X_train, y_train, X_test, y_test, confusion_matrix=True, normalizeCM=False, roc=True, plot_calibration=True, random_state=42, title="", pos_label=1):
    """Função auxiliar para execução de modelos de classificação.
    
    Parâmetros:
    
    - model: modelo de classificação a ser executado
    - X_train: base de treinamento das variáveis preditoras
    - y_train: base de treinamento da classe
    - X_test: base de teste das variáveis preditoras
    - y_test: base de teste da classe
    - confusion_matrix (default: True): exibir a matriz de confusão da classificação
    - normalizeCM (default: False): define se a matriz de confusão será normalizada
    - roc (default: True): define se será exibida a curva ROC para o classificador
    - plot_calibration (default: True): define se será exibida a curva de calibração para o classificador
    - title: define o título a ser exibido nos gráficos
    - pos_label: indica qual o valor de y_train e y_test que representa a classe positiva. O valor default é 1. 

    """
    clf = model
    name = title
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
        
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)
    else:  # usar decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    if confusion_matrix:
       skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=normalizeCM, title=name)
    if roc:
       skplt.metrics.plot_roc(y_test, prob_pos, plot_micro=False, plot_macro=False, classes_to_plot=[1], title=name,figsize=(10,10))
     
            
    prob_pos = prob_pos[:,1]
    clf_score = brier_score_loss(y_test, prob_pos, pos_label=pos_label)
    print("%s:" % name)
    print("\tBrier: %1.3f" % (clf_score))
    print("\tROC(AUC) %1.3f" % roc_auc_score(y_test, prob_pos))
    print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
    print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
    print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))
        
    if plot_calibration:
      
      fraction_of_positives, mean_predicted_value =                 calibration_curve(y_test, prob_pos, n_bins=10)
      plt.rcParams.update({'font.size': 22})
      plt.rc('legend',**{'fontsize':22})
      fig = plt.figure(3, figsize=(10, 10))
      ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
      ax2 = plt.subplot2grid((3, 1), (2, 0))
      ax1.plot([0, 1], [0, 1], "k:", label="Perfeitamente calibrado",)
      ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                     label="%s (%1.3f)" % (name, clf_score))

      ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                     histtype="step", lw=2)

      ax1.set_ylabel("Fração de positivos")
      ax1.set_ylim([-0.05, 1.05])
      ax1.legend(loc="lower right")
      ax1.set_title('Gráfico de Calibração  (reliability curve)')
      
      ax2.set_xlabel("Valor médio predito")
      ax2.set_ylabel("Quantidade")
      ax2.legend(loc="upper center", ncol=2)
      
      for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
             ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(22)
        
      for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(22)
      
      plt.tight_layout()
      plt.show()


# In[727]:


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

# In[21]:


import pyforest
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.metrics import accuracy_score
import lazypredict
from lazypredict.Supervised import LazyClassifier;


# In[22]:


clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
models


# In[23]:



from pycaret.classification import *

# configurar experimento do PyCaret
clf = setup(df, target='target')

# comparar modelos
best_model = compare_models()


# In[24]:


clf = setup(df, target='target')
models = compare_models(include=['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf', 'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 'lightgbm'], sort='AUC', fold=5, round=4)


# In[25]:


gbc_model = create_model('gbc')
predict_model(gbc_model)


# In[26]:


tune_model(gbc_model)


# ### Execução dos algoritmos de machine learning

# ### 12) Construção dos modelos
# 
# Obs: foram selecionados os seguintes modelos:
# * LGBMClassifer: 1o melhor acuracia  e baixo tempo de construção
# * XGBClassifier: 2 melhor AUC e baixo tempo de construção
# * Regresão Logística: 3o melhor AUC e baixo tempo de construção

# 

# In[760]:


from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# create a LinearSVC model with default parameters
model = LinearSVC()

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the test data
y_pred = model.predict(X_test)

# evaluate the performance of the model using classification report
print(classification_report(y_test, y_pred))


# In[761]:


from sklearn.metrics import roc_curve, auc

# obtendo os valores de decisão do modelo
y_score = model.decision_function(X_test)

# calculando a curva AUROC
fpr = {}
tpr = {}
roc_auc = {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score[:, i])
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
plt.title('LinearSVC')
plt.legend(loc="lower right")

# adicionando os valores de AUROC no gráfico
plt.text(0.5, 0.3, "AUROC Classe 0 (Abaixo) = {:.3f}".format(roc_auc[0]), ha='center', va='center', size=12, color='blue')
plt.text(0.3, 0.5, "AUROC Classe 1 (Dentro) = {:.3f}".format(roc_auc[1]), ha='center', va='center', size=12, color='red')
plt.text(0.6, 0.7, "AUROC Classe 2 (Acima) = {:.3f}".format(roc_auc[2]), ha='center', va='center', size=12, color='green')

plt.show()


# In[762]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# create a support vector classifier object
svc = SVC()

# define the parameter grid
param_grid = {'C': [0.1, 1, 10, 100],
              'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
              'gamma': ['scale', 'auto']}

# create a grid search object with 5-fold cross-validation
grid_search = GridSearchCV(svc, param_grid, cv=5)

# fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# print the best hyperparameters and corresponding score
print("Best hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# #### LGBMClassifier 

# In[731]:


import lightgbm as lgb

model = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42)

# treinando o modelo
model.fit(X_train, y_train)

# fazendo as previsões
y_pred = model.predict(X_test)

# avaliando o modelo
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[732]:


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

# In[733]:


import xgboost as xgb
model = xgb.XGBClassifier(objective='multi:softmax', num_class=3)

# treinando o modelo
model.fit(X_train, y_train)

# fazendo as previsões
y_pred = model.predict(X_test)

# avaliando o modelo
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[734]:


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

# In[735]:


from sklearn.linear_model import LogisticRegression

# Criando o modelo
model = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42)

# Treinando o modelo
model.fit(X_train, y_train)

# avaliando o modelo
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[736]:


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

# In[737]:


get_ipython().system('pip install boruta')


# In[738]:


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

# In[739]:


get_ipython().system('pip install boruta')


# In[740]:


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


# In[741]:


# Modelo com Boruta - Selecionado as variáveis preditoras
X_train_boruta = X_train[['a_pesoat', 'imc']]

X_test_boruta = X_test[['a_pesoat', 'imc']]

print('Shape sem boruta', X_train.shape, X_test.shape)
print('Shape com boruta', X_train_boruta.shape, X_test_boruta.shape)


# #### LGBM com Boruta

# In[742]:


#LGBM
clf_lgbm_boruta = LGBMClassifier()
clf_lgbm_boruta.fit(X_train_boruta, y_train)

y_pred_lgbm_boruta = clf_lgbm_boruta.predict(X_test_boruta)
prob_pos_lgbm_boruta = clf_lgbm_boruta.predict_proba(X_test_boruta)[:,1]

print('LGBM Model')
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_lgbm_boruta)))


print(classification_report(y_test, y_pred_lgbm_boruta))


# In[743]:


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

# In[744]:


# xgboost
clf_xgboost_boruta = xgb.XGBClassifier()
clf_xgboost_boruta.fit(X_train_boruta, y_train)

y_pred_xgboost_boruta = clf_xgboost_boruta.predict(X_test_boruta)
prob_pos_xgboost_boruta = clf_xgboost_boruta.predict_proba(X_test_boruta)[:,1]

print('XGBoost Model')
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_xgboost_boruta)))


print(classification_report(y_test, y_pred_xgboost_boruta))


# In[745]:


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

# In[746]:


# logistic regression
clf_lr_boruta = LogisticRegression()
clf_lr_boruta.fit(X_train_boruta, y_train)

y_pred_lr_boruta = clf_lr_boruta.predict(X_test_boruta)
prob_pos_lr_boruta = clf_lr_boruta.predict_proba(X_test_boruta)[:,1]

print('Logistic Regression Model')
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_lr_boruta)))


print(classification_report(y_test, y_pred_lr_boruta))


# In[747]:


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

# In[748]:


get_ipython().system('pip install shap')
import shap


# In[749]:


# resultados no teste
shap_values_test = shap.TreeExplainer(clf_xgboost_boruta).shap_values(X_test_boruta)
shap.summary_plot(shap_values_test, X_test_boruta, plot_type="bar")


# In[ ]:





# In[ ]:




