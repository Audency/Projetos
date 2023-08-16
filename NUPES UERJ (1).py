#!/usr/bin/env python
# coding: utf-8

# # Estimativas da prevalência de desnutrição e obesidade infantil nos municípios brasileiros: Alogaritmos de Machine Learnng
# 
# #### Audencio Victor, Ruchelli França de Lima e  Alexandre Chiavegatto Filho
# 
# ##### Dataset:Enani 
# 
# ##### Objetivo da Análise: Predizer a prevalência de desnutrição, obesidade  e  micronutrientes infantil nos municípios brasileiros
# 
# ##### O Conjunto de dados(ENANI)possui 1455 observações, com as as seguintes variáveis listadas:

# ### O Conjunto de dados possui 14560  criancas de 120 municipios, com as as seguintes variáveis listadas:
# 
#  1. id_desidentificado - Código de identificação da criança
#  2. a00_regiao-	Macrorregião 
#  3. a01_uf -Unidade da federação - seguindo os codigos do IBGE
#  4. a02_municipio -	Município - seguindo os codigos do IBGE
#  5. a11_situacao -Situação do domicílio
#  6. b02_sexo -Sexo da criança
#  7. b05_data-Data da coleta de dados do questionário geral
#  8. b05a_idade_em_meses-Idade da criança em meses no momento da coleta do questionario geral
#  9. vd_s00d_idade_peso- Idade da criança em meses no momento da coleta antropométrica
#  10. d01_cor -Raça/Cor da criança
#  11. p10_esgoto - Tipo de esgotamento sanitário
#  12. p11_agua - Tipo de fornecimento de água
#  13. p12_lixo- Tipo de destinação do lixo
#  14. ien_quintos -Indicador econômico nacional em quintos
#  15. h10_consulta- Local em que costuma levar a criança para atendimento médico
#  16. hb_final - Concentração sérica de hemoglobina (mg/dL)
#  17. anemia - Diagnóstico de anemia
#  18. vita_final- Concentração sérica de vitamina A (mg/dL)
#  19. vita_final_umol- Concentração sérica de vitamina A (mmol/L)
#  20. deficiencia_vita- Diagnóstico de deficiência de vitamina A
#  21. vd_zwaz- Score Z do indicador Peso para a idade
#  22. vd_zwaz_categ - Diagnóstico do estado nutricional pelo indicador Peso para a idade
#  23. vd_zimc - Score Z delo indicador índice de massa corporal para a idade
#  24. zimc_categ - Diagnóstico do estado nutricional pelo indicador índice de massa corporal para a idade
#  25. vd_zhaz- Score Z delo indicador estatura para a idade
#  26. zhaz_categ - Diagnóstico do estado nutricional pelo indicador estatura para a idade
#  27. vd_anthro_zwfl- Score Z delo indicador peso para estatura
#  28. anthro_zwfl_categ- Diagnóstico do estado nutricional pelo indicador peso para estatura
#  29. vd_idade_dias_ig- Idade gestacional ao nascimento + idade em dias
# 

# ## Roteiro:
# 
# 1. Limpeza e tratamento de dados(chamada do Banco)
# 2. Importação dos Pacotes
# 3. Transformação das Variáveis
# 4. Separação em treino e teste - Split aplicado de 80%
# 5. Retirada de variáveis indesejadas
# 6. Correlação/Teste de dependência 
# 7. Gráficos de frequências para X_train
# 8. One Hot Encoding
# 9. Treinamento do Primeiro Modelo 
# 10. Obtenção das métricas com base no conjunto de teste 
# 11. Treinando e comparando múltiplos algoritmos 
# 12. Desenvolvendo algoritmo selecionado 
# 13. Feature Selection - Boruta 
# 

# ### 1.Importação dos Pacotes

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

# In[3]:


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


# #### Carregar o banco de dados 

# In[4]:


# Lista com os nomes das colunas que você deseja importar
var = ['id_desidentificado','a00_regiao','a01_uf','a02_municipio', 'a11_situacao','b05a_idade_em_meses','anemia',
                     'deficiencia_vita', 'vd_zwaz_categ','vd_zimc_categ','vd_zhaz_categ'] 


# In[5]:


# Lê apenas as colunas desejadas no Dataset 
df = pd.read_excel("dataset_enani_final.xlsx", usecols=var)
df.head(20)


# In[6]:


# criando uma Varivel populatoat por minicipio 
df['pop'] = df['a02_municipio']
df.columns


# ### Transformar as categorias dos desfechos em numero computavel - Para calcular a prevalencia ###

# In[7]:


#### Desnutricao  por peso/idade 
a =df['vd_zwaz_categ'].unique()
print (a)

# Definir o dicionário de mapeamento
mapeamento = {'Peso adequado': 0, 'Peso elevado': 0, 'Baixo peso': 1, 'Muito baixo peso': 1}

# Recodificar a coluna 'vd_zwaz_categ' do DataFrame 'df' usando o método map()
df['vd_zwaz_categ'] = df['vd_zwaz_categ'].map(mapeamento)

# Mostrar o resultado
print(df['vd_zwaz_categ'].unique())


# In[8]:


# Calcular o  total de crincas desnutridas por Peso/idade
somatorio_desnutricao = df['vd_zwaz_categ'].sum()

# Mostrar o resultado
print("Total de crincas desnutridas por Peso/idade:", somatorio_desnutricao)


# #### Desnutricao  por Estatura/idade

# In[9]:


#### Desnutricao  por Estatura/idade 
b =df['vd_zhaz_categ'].unique()
print (b)

# Definir o dicionário de mapeamento
mapeamento = {'Altura adequada': 0, 'Baixa altura': 1, 'Muito baixa altura': 1}

# Recodificar a coluna 'vd_zwaz_categ' do DataFrame 'df' usando o método map()
df['vd_zhaz_categ'] = df['vd_zhaz_categ'].map(mapeamento)

# Mostrar o resultado
print(df['vd_zhaz_categ'].unique())


# In[10]:


# Calcular o  total de crincas desnutridas por Estatura/idade
somatorio_desnutricaop = df['vd_zhaz_categ'].sum()

# Mostrar o resultado
print("Total de crincas desnutridas por Estatura/idade:", somatorio_desnutricaop)


# #### Obesidade  por IMC/idade 

# In[11]:


c = df['vd_zimc_categ'].unique()
print(c)

# Definir o dicionário de mapeamento
mapeamento = {'Eutrofia': 0,'Magreza': 0,'Magreza acentuada': 0,'Risco de sobrepeso': 0,'Obesidade': 1,
              'Sobrepeso': 0}

# Recodificar a coluna 'vd_zwaz_categ' do DataFrame 'df' usando o método map()
df['vd_zimc_categ'] = df['vd_zimc_categ'].map(mapeamento)

# Mostrar o resultado
print(df['vd_zimc_categ'].unique())


# In[12]:


## Calcular o  total de crincas obesas pelo IMC para idade 
somatorio_obesida = df['vd_zimc_categ'].sum()

# Mostrar o resultado
print("Total de obesas:", somatorio_obesida)


# #### Viatamina A

# In[13]:


# # Calcular o  total de crincas com defiencia de Vita A 
somatorio_vita = df['deficiencia_vita'].sum()

# Mostrar o resultado
print("Total de crincas com defiencia de Vita A :", somatorio_vita )


# #### Anemia

# In[14]:


# # Calcular o total de crincas anemicas 
somatorio_anemia = df['anemia'].sum()

# Mostrar o resultado
print("Total de crincas anemicas :", somatorio_anemia)


# #### Transformando a variavel idade em 2 grupos: ( O = abaixo de 24 meses e 1 = acima de 24 meses)

# In[15]:


import re

# Passo 1: Converter a coluna 'b05a_idade_em_meses' para strings e extrair apenas os números usando expressões regulares
df['b05a_idade_em_meses'] = df['b05a_idade_em_meses'].astype(str).str.extract(r'(\d+)').astype(float)

# Passo 2: Criar uma nova coluna 'idade_classificada' com valores 0 ou 1
df['b05a_idade_em_meses'] = df['b05a_idade_em_meses'].apply(lambda idade: 1 if idade >= 25 else 0)

# Exibir o DataFrame resultante
print(df)


# In[16]:


df.head()


# In[17]:


df.info() # tomando informações sobre o tipo das variáveis parametrizadas na chamada


# In[18]:


df.tail
df.columns  # para ver os nomes das  variveis 
list(df.columns.values.tolist()) # listar todas as variaveis 


# In[19]:


## Numero toral de municipios 
v = df['a02_municipio'].nunique()
print("Número de municípios:", v)


# #### Organizacao do banco e transformacao para o agregado 

# In[20]:


# Agregar novamente apenas por municípios para obter os totais para cada município

df_agre = df.groupby(['a02_municipio','a00_regiao','a01_uf']).agg({
    'anemia': 'sum',
    'deficiencia_vita': 'sum',
    'vd_zimc_categ': 'sum',
    'vd_zhaz_categ': 'sum',
    'vd_zwaz_categ': 'sum',
    'pop': 'count'  # Contagem total de ocorrências por município
}).reset_index()


# In[21]:


df_agre.head()


# In[22]:


df_agre.info()
## Numero toral de municipios 


# #### Calcular a prevalência dos desfechos  (Desnutricao, Obesidade, Viatamina A e Anemia)

# In[23]:


# Calcular a prevalência de Desnuticao por município (Zcore de Altura para idade)
df_agre['Prev_desnut'] = df_agre['vd_zhaz_categ'] / df_agre ['pop']*100

# Calcular a prevalência de Desnuticao por município (Zcore de Peso para idade)
df_agre['Prev_desnut1'] = df_agre['vd_zwaz_categ'] / df_agre ['pop']*100

# Calcular a prevalência de Obesidade por município
df_agre['Prev_Obesi'] = df_agre['vd_zimc_categ'] / df_agre ['pop']*100

# Calcular a prevalência de anemia por município
df_agre['Prev_anem'] = df_agre['anemia'] / df_agre ['pop']*100

# Calcular a prevalência de Viatamina A por município
df_agre['Prev_viaa'] = df_agre['deficiencia_vita'] / df_agre ['pop']*100


# In[24]:


print(df_agre)


# In[25]:


df_agre.info()


# ###### As analises desccritivas no Banco de dados 

# In[26]:


# Agrupar por região e calcular a média da prevalência de cada desfecho

prevalencia_regiao = df_agre.groupby('a00_regiao').mean()[['Prev_desnut', 'Prev_desnut1','Prev_Obesi', 'Prev_anem', 'Prev_viaa']]

print(prevalencia_regiao)


# In[27]:


# Calcular a média geral da prevalência de cada desfecho
prevalencia_geral = df_agre.mean()[['Prev_desnut', 'Prev_desnut1', 'Prev_Obesi', 'Prev_anem', 'Prev_viaa']]

print("Prevalência geral:")
print(prevalencia_geral)


# In[28]:


import matplotlib.pyplot as plt

# Dados de exemplo
desfechos = prevalencia_geral.index
valores = prevalencia_geral.values

# Definir uma paleta de cores agradável
cores = plt.cm.Set2(range(len(desfechos)))

# Aumentar o tamanho do gráfico
plt.figure(figsize=(10, 6))

# Plotar o gráfico de barras com cores definidas e rótulos nos eixos
plt.bar(desfechos, valores, color=cores, edgecolor='black', linewidth=1.5)
plt.xlabel('Desfechos', fontsize=12)
plt.ylabel('Prevalência Média', fontsize=12)
plt.title('Média Geral da Prevalência de Cada Desfecho', fontsize=16, fontweight='bold')

# Adicionar os valores acima das barras com duas casas decimais
for i in range(len(desfechos)):
    valor_formatado = "{:.2f}".format(valores[i])
    plt.text(desfechos[i], valores[i], valor_formatado, ha='center', va='bottom', fontsize=11, fontweight='bold')

# Girar os rótulos do eixo x e ajustar o espaçamento
plt.xticks(rotation=45, ha='right', fontsize=11)

# Remover bordas do gráfico
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Adicionar uma grade de fundo
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Ajustar os limites do eixo y para melhorar a visualização dos valores
plt.ylim(0, max(valores) * 1.1)

# Mostrar o gráfico
plt.tight_layout()
plt.show()


# In[29]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.boxplot([df_agre['anemia'], df_agre['deficiencia_vita'], df_agre['vd_zimc_categ'], df_agre['vd_zhaz_categ'], df_agre['vd_zwaz_categ']],
            labels=['Anemia', 'Deficiência de Vitamina A', 'Desnutrição ZIMC', 'Desnutrição ZHAZ', 'Desnutrição ZWAZ'])
plt.title('Boxplot das Prevalências dos Desfechos')
plt.ylabel('Prevalência')

plt.show()


# In[30]:



# Agrupar por região e calcular a média da prevalência de cada desfecho
prevalencia_regiao = df_agre.groupby('a00_regiao').mean()[['Prev_desnut','Prev_desnut1',  'Prev_Obesi', 'Prev_anem', 'Prev_viaa']]

# Configurações do gráfico
plt.figure(figsize=(10, 6))
bar_width = 0.2
index = range(len(prevalencia_regiao))

# Gráfico de barras para cada desfecho
plt.bar(index, prevalencia_regiao['Prev_desnut'], width=bar_width, label='Desnutrição')
plt.bar([i + bar_width for i in index], prevalencia_regiao['Prev_desnut1'], width=bar_width, label='Desnutrição1')
plt.bar([i + 2*bar_width for i in index], prevalencia_regiao['Prev_Obesi'], width=bar_width, label='Obesidade')
plt.bar([i + 3*bar_width for i in index], prevalencia_regiao['Prev_anem'], width=bar_width, label='Anemia')
plt.bar([i + 4*bar_width for i in index], prevalencia_regiao['Prev_viaa'], width=bar_width, label='Deficiência de Vitamina A')


# Configurações adicionais do gráfico
plt.xlabel('Região')
plt.ylabel('Prevalência Média')
plt.title('Prevalência dos Desfechos por Região')
plt.xticks([i + 1.5*bar_width for i in index], prevalencia_regiao.index)
plt.legend()

# Exibir o gráfico
plt.tight_layout()
plt.show()


# ##  outlier 

# In[31]:



# Função para identificar outliers usando o método IQR
def identificar_outliers(dados, coluna):
    Q1 = dados[coluna].quantile(0.25)
    Q3 = dados[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = dados[(dados[coluna] < limite_inferior) | (dados[coluna] > limite_superior)]
    return outliers

# Identificando outliers em cada desfecho
outliers_desnutricao = identificar_outliers(df_agre, 'Prev_desnut')
outliers_desnutricao1 = identificar_outliers(df_agre, 'Prev_desnut1')
outliers_obesidade = identificar_outliers(df_agre, 'Prev_Obesi')
outliers_anemia = identificar_outliers(df_agre, 'Prev_anem')
outliers_vitamina_a = identificar_outliers(df_agre, 'Prev_viaa')

# Exibindo os outliers e suas respectivas unidades federativas para cada desfecho
print("Outliers de Desnutrição E/I:")
print(outliers_desnutricao[['a02_municipio', 'a01_uf', 'Prev_desnut']])
print("Outliers de Desnutrição P/I:")
print(outliers_desnutricao1[['a02_municipio', 'a01_uf', 'Prev_desnut1']])
print("\nOutliers de Obesidade:")
print(outliers_obesidade[['a02_municipio', 'a01_uf', 'Prev_Obesi']])
print("\nOutliers de Anemia:")
print(outliers_anemia[['a02_municipio', 'a01_uf', 'Prev_anem']])
print("\nOutliers de Deficiência de Vitamina A:")
print(outliers_vitamina_a[['a02_municipio', 'a01_uf', 'deficiencia_vita']])


# In[32]:


## Removendo os outlier no dataset
df_agre=df_agre[(df_agre['Prev_desnut'] < 23.0)]


# In[33]:


#df_agre=df_agre[(df_agre['Prev_desnut1'] < 7.9)]


# In[34]:


#df_agre=df_agre[(df_agre['Prev_Obesi'] < 9.6)]


# In[35]:


#df_agre= df_agre[(df_agre['Prev_anem'] < 10.7)]


# In[36]:


#df_agre=df_agre[(df_agre['deficiencia_vita'] < 2.0)]


# In[37]:


df_agre.head(20)


# In[38]:


# Remover municípios com valores abaixo de 0.5 de prevalência da desnutrição
df_agrea = df_agre[df_agre['Prev_desnut'] >= 0.5]


# In[39]:


df_agrea.head(20)


# In[40]:


df_agre.info()


# In[41]:


# Criar gráfico de boxplot para cada desfecho
plt.figure(figsize=(10, 6))
plt.boxplot([df_agre['Prev_desnut'], df_agre['Prev_desnut1'], df_agre['Prev_Obesi'], df_agre['Prev_anem'], df_agre['deficiencia_vita']],
            labels=['Desnutrição E/I', 'Desnutrição P/I', 'Obesidade', 'Anemia', 'Deficiência de Vitamina A'],
            sym='r+')  # destaca os outliers com símbolo vermelho '+'

plt.xlabel('Desfecho')
plt.ylabel('Valores')
plt.title('Gráfico de Boxplot para Outliers')
plt.xticks(rotation=45)
plt.grid(True)

# Exibir o gráfico
plt.tight_layout()
plt.show()


# ### Estratificando por Idade das criancas 

# In[42]:


# Agregar novamente apenas por municípios para obter os totais para cada município

df_agre1 = df.groupby(['a02_municipio', 'b05a_idade_em_meses','a00_regiao','a01_uf']).agg({
    'anemia': 'sum',
    'deficiencia_vita': 'sum',
    'vd_zimc_categ': 'sum',
    'vd_zhaz_categ': 'sum',
    'vd_zwaz_categ': 'sum',
    'pop': 'count'  # Contagem total de ocorrências por município
}).reset_index()


# ## O a 24 meses 

# In[43]:


df_agre0 = df_agre1 [df_agre1['b05a_idade_em_meses']== 0]
df_agre0.head()


# In[44]:


##Calcular a prevalência dos desfechos (Desnutricao, Obesidade, Viatamina A e Anemia)

# Calcular a prevalência de Desnuticao por município (Zcore de Altura para idade)
df_agre0['Prev_desnut'] = df_agre0['vd_zhaz_categ'] / df_agre0 ['pop']*100

# Calcular a prevalência de Desnuticao por município (Zcore de Peso para idade)
df_agre0['Prev_desnut1'] = df_agre0['vd_zwaz_categ'] / df_agre0 ['pop']*100

# Calcular a prevalência de Obesidade por município
df_agre0['Prev_Obesi'] = df_agre0['vd_zimc_categ'] / df_agre0 ['pop']*100

# Calcular a prevalência de anemia por município
df_agre0['Prev_anem'] = df_agre0['anemia'] / df_agre0 ['pop']*100

# Calcular a prevalência de Viatamina A por município
df_agre0['Prev_viaa'] = df_agre0['deficiencia_vita'] / df_agre0 ['pop']*100


# In[45]:


print(df_agre0)


# In[46]:


# Agrupar por região e calcular a média da prevalência de cada desfecho

prevalencia_regiao0 = df_agre0.groupby('a00_regiao').mean()[['Prev_desnut', 'Prev_desnut1','Prev_Obesi', 'Prev_anem', 'Prev_viaa']]

print(prevalencia_regiao0)


# In[47]:


# Calcular a média geral da prevalência de cada desfecho
prevalencia_geral0 = df_agre0.mean()[['Prev_desnut', 'Prev_desnut1', 'Prev_Obesi', 'Prev_anem', 'Prev_viaa']]

print("Prevalência geral0:")
print(prevalencia_geral0)


# In[48]:


# Função para identificar outliers usando o método IQR
def identificar_outliers(dados, coluna):
    Q1 = dados[coluna].quantile(0.25)
    Q3 = dados[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = dados[(dados[coluna] < limite_inferior) | (dados[coluna] > limite_superior)]
    return outliers

# Identificando outliers em cada desfecho
outliers_desnutricao = identificar_outliers(df_agre0, 'Prev_desnut')
outliers_desnutricao1 = identificar_outliers(df_agre0, 'Prev_desnut1')
outliers_obesidade = identificar_outliers(df_agre0, 'Prev_Obesi')
outliers_anemia = identificar_outliers(df_agre0, 'Prev_anem')
outliers_vitamina_a = identificar_outliers(df_agre0, 'Prev_viaa')

# Exibindo os outliers e suas respectivas unidades federativas para cada desfecho
print("Outliers de Desnutrição E/I:")
print(outliers_desnutricao[['a02_municipio', 'a01_uf', 'Prev_desnut']])
print("Outliers de Desnutrição P/I:")
print(outliers_desnutricao1[['a02_municipio', 'a01_uf', 'Prev_desnut1']])
print("\nOutliers de Obesidade:")
print(outliers_obesidade[['a02_municipio', 'a01_uf', 'Prev_Obesi']])
print("\nOutliers de Anemia:")
print(outliers_anemia[['a02_municipio', 'a01_uf', 'Prev_anem']])
print("\nOutliers de Deficiência de Vitamina A:")
print(outliers_vitamina_a[['a02_municipio', 'a01_uf', 'deficiencia_vita']])


# ## Criancas de 24 a 59 meses

# In[49]:


df_agre1 = df_agre1[df_agre1['b05a_idade_em_meses'] == 1]
df_agre1.head()


# In[50]:


##Calcular a prevalência dos desfechos (Desnutricao, Obesidade, Viatamina A e Anemia)

# Calcular a prevalência de Desnuticao por município (Zcore de Altura para idade)
df_agre1['Prev_desnut'] = df_agre1['vd_zhaz_categ'] / df_agre1 ['pop']*100

# Calcular a prevalência de Desnuticao por município (Zcore de Peso para idade)
df_agre1['Prev_desnut1'] = df_agre1['vd_zwaz_categ'] / df_agre1 ['pop']*100

# Calcular a prevalência de Obesidade por município
df_agre1['Prev_Obesi'] = df_agre1['vd_zimc_categ'] / df_agre1 ['pop']*100

# Calcular a prevalência de anemia por município
df_agre1['Prev_anem'] = df_agre1['anemia'] / df_agre1 ['pop']*100

# Calcular a prevalência de Viatamina A por município
df_agre1['Prev_viaa'] = df_agre1['deficiencia_vita'] / df_agre1 ['pop']*100


# In[51]:


print(df_agre1)


# In[52]:


# Agrupar por região e calcular a média da prevalência de cada desfecho

prevalencia_regiao1 = df_agre1.groupby('a00_regiao').mean()[['Prev_desnut', 'Prev_desnut1','Prev_Obesi', 'Prev_anem', 'Prev_viaa']]

print(prevalencia_regiao1)


# In[53]:


# Calcular a média geral da prevalência de cada desfecho
prevalencia_geral1 = df_agre1.mean()[['Prev_desnut', 'Prev_desnut1', 'Prev_Obesi', 'Prev_anem', 'Prev_viaa']]

print("Prevalência geral1:")
print(prevalencia_geral1)


# In[54]:



# Função para identificar outliers usando o método IQR
def identificar_outliers(dados, coluna):
    Q1 = dados[coluna].quantile(0.25)
    Q3 = dados[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = dados[(dados[coluna] < limite_inferior) | (dados[coluna] > limite_superior)]
    return outliers

# Identificando outliers em cada desfecho
outliers_desnutricao = identificar_outliers(df_agre1, 'Prev_desnut')
outliers_desnutricao1 = identificar_outliers(df_agre1, 'Prev_desnut1')
outliers_obesidade = identificar_outliers(df_agre1, 'Prev_Obesi')
outliers_anemia = identificar_outliers(df_agre1, 'Prev_anem')
outliers_vitamina_a = identificar_outliers(df_agre1, 'Prev_viaa')

# Exibindo os outliers e suas respectivas unidades federativas para cada desfecho
print("Outliers de Desnutrição E/I:")
print(outliers_desnutricao[['a02_municipio', 'a01_uf', 'Prev_desnut']])
print("Outliers de Desnutrição P/I:")
print(outliers_desnutricao1[['a02_municipio', 'a01_uf', 'Prev_desnut1']])
print("\nOutliers de Obesidade:")
print(outliers_obesidade[['a02_municipio', 'a01_uf', 'Prev_Obesi']])
print("\nOutliers de Anemia:")
print(outliers_anemia[['a02_municipio', 'a01_uf', 'Prev_anem']])
print("\nOutliers de Deficiência de Vitamina A:")
print(outliers_vitamina_a[['a02_municipio', 'a01_uf', 'deficiencia_vita']])


# ### Organizar as variveis com as preditoras 

# In[55]:


dfn = pd.read_stata("banco.dta")
colunas_selecionadas =['CodMunicp','UF','Municip', 'População','PortMunicip','pop_rural', 'pop_urbana',
                       'RendDomiciliar',  'Densestasaudaveis', 'DensestaNaosaudaveis', 'idhm', 'idhm_renda', 
                       'idhm_longevidade', 'idhm_educacao', 'pibpcapita', 'renda', 'renda_gini', 'analfa_total',
                       'analfa_mulher', 'esc_anos','desemprego', 'pobres', 'extr_pobres',  'CoberESF1', 'CoberAB1']
dfn = dfn[colunas_selecionadas].copy()

# renomeando a variavel CodMunicp  para  a02_municipio
dfn = dfn.rename(columns={'CodMunicp': 'a02_municipio'})
dfn.head()


# ### Fazer linkedge com as variaves Peditoras 

# In[550]:


# Convert 'a02_municipio' column to string in both DataFrames
dfn['a02_municipio'] = dfn['a02_municipio'].astype(str)
df_agre['a02_municipio'] = df_agre['a02_municipio'].astype(str)

# Now, perform the merge operation
df_merge = dfn.merge(df_agre, on='a02_municipio', how='left')


# In[ ]:


# APagar municipios sem informacao  no desfecho
df_merge = dfn.merge(df_agre, on='a02_municipio', how='left')
df_merge.dropna(subset=['Prev_desnut', 'Prev_desnut1', 'Prev_Obesi', 'Prev_anem', 'Prev_viaa'], inplace=True)


# In[ ]:


df_merge.head()


# In[ ]:


df_merge.info() 


# In[ ]:


#Covertentendo a variaval Populacao  e Regiao  object  de para Flot (quantitativa)
df_merge['População'] = df_merge['População'].str.replace('.', '').astype(float)


# In[ ]:


pd.set_option('display.max_columns', None)
# Print the DataFrame
print(df_merge)


# In[ ]:


# Agrupar por região e calcular a média da prevalência de cada desfecho

prevalencia_regiao = df_merge.groupby('a00_regiao').mean()[['Prev_desnut', 'Prev_desnut1','Prev_Obesi', 'Prev_anem', 'Prev_viaa']]

print(prevalencia_regiao)


# In[ ]:


# Agrupar por região e calcular a média da prevalência de cada desfecho

prevalencia_estado = df_merge.groupby(['UF','a00_regiao']).mean()[['Prev_desnut', 'Prev_desnut1','Prev_Obesi', 'Prev_anem', 'Prev_viaa']]
prevalencia_estado.sort_values(by='a00_regiao', ascending=True, inplace=True)
print(prevalencia_estado)


# In[ ]:



# Função para identificar outliers usando o método IQR
def identificar_outliers(dados, coluna):
    Q1 = dados[coluna].quantile(0.25)
    Q3 = dados[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = dados[(dados[coluna] < limite_inferior) | (dados[coluna] > limite_superior)]
    return outliers

# Identificando outliers em cada desfecho
outliers_desnutricao = identificar_outliers(df_merge, 'Prev_desnut')
outliers_desnutricao1 = identificar_outliers(df_merge, 'Prev_desnut1')
outliers_obesidade = identificar_outliers(df_merge, 'Prev_Obesi')
outliers_anemia = identificar_outliers(df_merge, 'Prev_anem')
outliers_vitamina_a = identificar_outliers(df_merge, 'Prev_viaa')

# Exibindo os outliers e suas respectivas unidades federativas para cada desfecho
print("Outliers de Desnutrição E/I:")
print(outliers_desnutricao[['UF','a00_regiao','Municip','Prev_desnut']])
print("Outliers de Desnutrição P/I:")
print(outliers_desnutricao1[['UF','a00_regiao','Municip', 'Prev_desnut1']])
print("\nOutliers de Obesidade:")
print(outliers_obesidade[['UF','a00_regiao','Municip', 'Prev_Obesi']])
print("\nOutliers de Anemia:")
print(outliers_anemia[['UF','a00_regiao','Municip', 'Prev_anem']])
print("\nOutliers de Deficiência de Vitamina A:")
print(outliers_vitamina_a[['UF','a00_regiao','Municip', 'deficiencia_vita']])


# In[ ]:


# Mudando o nomde do dataset 
df = df_merge
df.head()


# In[ ]:


df.info()


# In[ ]:


df.isna()
df.isna().sum()


# In[ ]:


df.drop(['deficiencia_vita', 'vd_zimc_categ','anemia', 'vd_zhaz_categ', 'vd_zwaz_categ','pop'], axis=1, inplace=True)


# In[ ]:


# Calcular a matriz de correlação
correlation_matrix = df.corr()

print(correlation_matrix)


# In[ ]:


# Calcular a matriz de correlação com o desfecho
correlation_matrix = df.corr()

# Criar o mapa de calor
plt.figure(figsize=(25, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Exibir o gráfico
plt.title('Correlação com o Desfecho')
plt.show()


# In[ ]:


plt.figure(figsize=(8,12))
sns.heatmap(df.corr(), annot=True, linecolor='black', linewidths=1, cmap='jet')
plt.show()


# In[ ]:


# Calcular a matriz de correlação com o desfecho
correlation_matrix = df.corr()['Prev_desnut']

plt.figure(figsize=(12, 8))
sns.barplot(x=correlation_matrix.index, y=correlation_matrix.values)

# Exibir o gráfico
plt.title('Correlação com o Desnitricao Estatura/Idade')
plt.xlabel('Variável')
plt.ylabel('Correlação')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# Calcular a matriz de correlação com o desfecho
correlation_matrix = df.corr()['Prev_desnut1']

plt.figure(figsize=(12, 8))
sns.barplot(x=correlation_matrix.index, y=correlation_matrix.values)

# Exibir o gráfico
plt.title('Correlação com o Desnitricao Peso/idade')
plt.xlabel('Variável')
plt.ylabel('Correlação')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# Calcular a matriz de correlação com o desfecho
correlation_matrix = df.corr()['Prev_Obesi']

plt.figure(figsize=(12, 8))
sns.barplot(x=correlation_matrix.index, y=correlation_matrix.values)

# Exibir o gráfico
plt.title('Correlação com o Obesidade')
plt.xlabel('Variável')
plt.ylabel('Correlação')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# Calcular a matriz de correlação com o desfecho
correlation_matrix = df.corr()['Prev_anem']

plt.figure(figsize=(12, 8))
sns.barplot(x=correlation_matrix.index, y=correlation_matrix.values)

# Exibir o gráfico
plt.title('Correlação com o Prevalancia de Anemia')
plt.xlabel('Variável')
plt.ylabel('Correlação')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# Calcular a matriz de correlação com o desfecho
correlation_matrix = df.corr()['Prev_viaa']

plt.figure(figsize=(12, 8))
sns.barplot(x=correlation_matrix.index, y=correlation_matrix.values)

# Exibir o gráfico
plt.title('Correlação com o Prevalancia de Vitamina A')
plt.xlabel('Variável')
plt.ylabel('Correlação')
plt.xticks(rotation=90)
plt.show()


# #### Automatizar a  Analise descritiva  dos dados no Banco 

# In[644]:


pip install pandas-profiling


# In[645]:


import ydata_profiling
ydata_profiling.ProfileReport(df)


# In[ ]:


import pandas as pd
from pandas_profiling import ProfileReport

# Assuming 'df' is your DataFrame
profile = ProfileReport(df)
print(profile)


# In[ ]:


profile.to_widgets()


# In[ ]:


profile.to_file("report.html")


# In[646]:


df.columns


# In[647]:


#Apagando variaveis do dataset 
df.drop(['a02_municipio', 'UF','PortMunicip','Municip','a00_regiao', 'Prev_desnut', 'Prev_desnut1',
         'Prev_anem','a01_uf', 'Prev_viaa'], axis=1, inplace=True)


# In[ ]:


# Para as variáveis categóricas iremos criar dummies/ One hotencode
#df= pd.get_dummies(df, columns=['a00_regiao'])


# In[649]:


##imputar dados pela media dsa variavel (Separar entre treino e teste)
df.fillna(df.mean(), inplace=True)


# In[650]:


# variável de interesse/desfecho 
target= df >> select('Prev_Obesi')


# In[651]:


#Defenindo as variveis preditoras

variaveis_preditoras = df.iloc[:, df.columns != 'Prev_Obesi']
classe = df.iloc[:, df.columns == 'Prev_Obesi']
X_train, X_test, y_train, y_test = train_test_split(variaveis_preditoras, 
                                                    classe,
                                                    train_size = 0.5,
                                                    random_state = 45)


# ###### Validacao cruzada

# In[652]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

# Carregar seus dados e dividir entre features (X) e rótulos (y)
# X = seus_dados_de_features
# y = seus_rótulos

# Criar o modelo de regressão que você deseja avaliar
model = LinearRegression()  # Substitua pelo modelo de regressão desejado

# Definir o número de folds para a validação cruzada
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Realizar a validação cruzada e obter as pontuações de desempenho
scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')

# Como o neg_mean_squared_error é utilizado, multiplicamos as pontuações por -1 para obter os MSEs positivos
mse_scores = -1 * scores

# Imprimir as pontuações de cada fold e a média das pontuações (MSEs)
for fold_num, mse_score in enumerate(mse_scores, start=1):
    print(f"Fold {fold_num}: Mean Squared Error = {mse_score:.2f}")

print(f"Mean MSE: {np.mean(mse_scores):.2f}")


# In[653]:


X_train.shape


# In[654]:


X_test.shape


# In[655]:


X_test.columns


# In[656]:


X_train.columns


# In[657]:


# Standarscaler com passthrough tem um problema de ordenação das colunas. Quando aplicamos, ele fornce o resultado com as colunas padronizadas em primeiro, seguidas das demais colunas.
# Para resolver este problema, iremos ordenar as nossas colunas alocando as contínuas nas primeiras posições 

X_train = X_train.loc[:,[ 'População', 'pop_rural', 'pop_urbana', 'RendDomiciliar',
       'Densestasaudaveis', 'DensestaNaosaudaveis', 'idhm', 'idhm_renda',
       'idhm_longevidade', 'idhm_educacao', 'pibpcapita', 'renda',
       'renda_gini', 'analfa_total', 'analfa_mulher', 'esc_anos', 'desemprego',
       'pobres', 'extr_pobres', 'CoberESF1', 'CoberAB1']]

X_test = X_test.loc[:,[ 'População', 'pop_rural', 'pop_urbana', 'RendDomiciliar',
       'Densestasaudaveis', 'DensestaNaosaudaveis', 'idhm', 'idhm_renda',
       'idhm_longevidade', 'idhm_educacao', 'pibpcapita', 'renda',
       'renda_gini', 'analfa_total', 'analfa_mulher', 'esc_anos', 'desemprego',
       'pobres', 'extr_pobres', 'CoberESF1', 'CoberAB1' ]]

X_train_columns = X_train.columns
X_test_columns = X_test.columns


# In[658]:


from sklearn.compose import ColumnTransformer

### variáveis contínuas que serão padronizadas
continuous_cols = ['População', 'pop_rural', 'pop_urbana', 'RendDomiciliar',
       'Densestasaudaveis', 'DensestaNaosaudaveis', 'idhm', 'idhm_renda',
       'idhm_longevidade', 'idhm_educacao', 'pibpcapita', 'renda',
       'renda_gini', 'analfa_total', 'analfa_mulher', 'esc_anos', 'desemprego',
       'pobres', 'extr_pobres', 'CoberESF1', 'CoberAB1']

def setScaler():
  ct = ColumnTransformer([
        ('scaler', StandardScaler(), continuous_cols)
    ], remainder='passthrough' # utilizamos para manter as colunas em que não aplicamos o scaler
  )
  return ct
  
scaler = setScaler()


# In[659]:


scaler.fit(X_train)


# In[660]:


X_train = scaler.transform(X_train)


# In[661]:


X_test = scaler.transform(X_test)


# In[662]:


# para evitarmos a exibição dos dados em notacao científica
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# In[663]:


X_train = pd.DataFrame(X_train, columns=X_train_columns)
X_test = pd.DataFrame(X_test, columns=X_test_columns)


# In[664]:


X_train.head()


# In[665]:


X_test.describe()


# In[666]:


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


# In[667]:


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


# In[668]:


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


# In[669]:


import xgboost as xgb
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Create and fit the XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Explain the model's predictions using SHAP values
explainer = shap.Explainer(model)
shap_values_train = explainer.shap_values(X_train)
shap_values_test = explainer.shap_values(X_test)

# Calculate the predicted values
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Create a DataFrame with SHAP values and predicted values for training set
shap_df_train = pd.DataFrame(shap_values_train, columns=X_train.columns)
shap_df_train['Predicted'] = y_pred_train

# Create a DataFrame with SHAP values and predicted values for test set
shap_df_test = pd.DataFrame(shap_values_test, columns=X_test.columns)
shap_df_test['Predicted'] = y_pred_test

# Plot the SHAP summary plot for training set
shap.summary_plot(shap_values_train, X_train, feature_names=X_train.columns)

# Plot the SHAP summary plot for test set
shap.summary_plot(shap_values_test, X_test, feature_names=X_test.columns)


# In[681]:


# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, color='black')
plt.xlabel('Prevalência Real')
plt.ylabel('Prevalência Prevista')
plt.title('Prevalência Real e Prevista da Obesidade Infantil por Município (Teste)')
plt.show()


# In[671]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calculate metrics for training set
train_mse = mean_squared_error(y_train, y_pred_train)
train_rmse = train_mse ** 0.5
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)

# Calculate metrics for test set
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = test_mse ** 0.5
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print("Metrics for Training Set:")
print(f"Mean Squared Error (MSE): {train_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {train_rmse:.2f}")
print(f"Mean Absolute Error (MAE): {train_mae:.2f}")
print(f"R-squared (R2) Score: {train_r2:.2f}")
print("\nMetrics for Test Set:")
print(f"Mean Squared Error (MSE): {test_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {test_rmse:.2f}")
print(f"Mean Absolute Error (MAE): {test_mae:.2f}")
print(f"R-squared (R2) Score: {test_r2:.2f}")


# In[672]:


importances = model.feature_importances_
feature_names = X_train.columns

# Sort the feature importances in descending order
sorted_indices = importances.argsort()[::-1]
sorted_importances = importances[sorted_indices]
sorted_features = feature_names[sorted_indices]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_importances)), sorted_importances, tick_label=sorted_features, color='indigo')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.xticks(rotation=45)
plt.show()


# In[ ]:





# In[349]:


import pyforest
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


# In[91]:


reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Print the results
print(models)


# In[378]:


pip install boruta


# In[379]:


pip install xgboost


# In[676]:


from boruta import BorutaPy
from xgboost import XGBRegressor  # Corrigido para importar XGBRegressor corretamente
import numpy as np

# Criando o estimador para o Boruta
forest = XGBRegressor(
   n_jobs=-1,
   max_depth=5
)
boruta = BorutaPy(
   estimator=forest,
   n_estimators='auto',
   max_iter=100  # Número de tentativas a serem realizadas
)

# Parametrizando para o conjunto de treino
boruta.fit(np.array(X_train), np.array(y_train))

# Resultados
green_area = X_train.columns[boruta.support_].to_list()
blue_area = X_train.columns[boruta.support_weak_].to_list()
print('Features na área verde:', green_area)
print('Features na área azul:', blue_area)


# # 14) Modelo de Xboost com Boruta

# In[677]:


# Modelo com Boruta
X_train_boruta = X_train[['Densestasaudaveis', 'analfa_total', 'esc_anos', 'pobres']]
X_test_boruta = X_test[['Densestasaudaveis', 'analfa_total', 'esc_anos', 'pobres']]

print(X_train_boruta.shape, X_test_boruta.shape)


# In[678]:


import xgboost as xgb
import shap
import pandas as pd

X_train_boruta = X_train[['Densestasaudaveis', 'analfa_total', 'esc_anos', 'pobres']]
X_test_boruta = X_test[['Densestasaudaveis', 'analfa_total', 'esc_anos', 'pobres']]

print(X_train_boruta.shape, X_test_boruta.shape)

model_boruta = xgb.XGBRegressor()
model_boruta.fit(X_train_boruta, y_train)

# Create and fit the XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train_boruta, y_train)

# Explain the model's predictions using SHAP values
explainer = shap.Explainer(model)
shap_values_train = explainer.shap_values(X_train_boruta)
shap_values_test = explainer.shap_values(X_test_boruta)

# Calculate the predicted values
y_pred_train = model.predict(X_train_boruta)
y_pred_test = model.predict(X_test_boruta)

# Create a DataFrame with SHAP values and predicted values for the training set
shap_df_train = pd.DataFrame(shap_values_train, columns=X_train_boruta.columns)
shap_df_train['Predicted'] = y_pred_train

# Create a DataFrame with SHAP values and predicted values for the test set
shap_df_test = pd.DataFrame(shap_values_test, columns=X_test_boruta.columns)
shap_df_test['Predicted'] = y_pred_test

# Plot the SHAP summary plot for the training set
shap.summary_plot(shap_values_train, X_train_boruta, feature_names=X_train_boruta.columns)

# Plot the SHAP summary plot for the test set
shap.summary_plot(shap_values_test, X_test_boruta, feature_names=X_test_boruta.columns)


# In[679]:




# Plot para conjunto de teste
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color='black', alpha=0.7, label='Prevalência da Desnutrição (E/I) por Município')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label='Melhor modelo')
plt.xlabel('Prevalência Real ')
plt.ylabel('Prevalência Prevista')
plt.title('Prevalência Real  e Prevista da Desnutrição por Município (Teste')
plt.legend()
plt.grid(True)
plt.show()


# In[680]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calcular métricas para o conjunto de treinamento
train_mse = mean_squared_error(y_train, y_pred_train)
train_rmse = train_mse ** 0.5
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)

# Calcular métricas para o conjunto de teste
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = test_mse ** 0.5
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print("Métricas para o Conjunto de Treinamento:")
print(f"Erro Quadrático Médio (MSE): {train_mse:.2f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {train_rmse:.2f}")
print(f"Erro Médio Absoluto (MAE): {train_mae:.2f}")
print(f"Coeficiente de Determinação (R2): {train_r2:.2f}")
print("\nMétricas para o Conjunto de Teste:")
print(f"Erro Quadrático Médio (MSE): {test_mse:.2f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {test_rmse:.2f}")
print(f"Erro Médio Absoluto (MAE): {test_mae:.2f}")
print(f"Coeficiente de Determinação (R2): {test_r2:.2f}")


# In[514]:


import pyforest
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


# In[515]:


reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train_boruta, X_test, y_train, y_test)

# Print the results
print(models)


# In[ ]:




