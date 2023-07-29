#!/usr/bin/env python
# coding: utf-8

# # Estimativas da prevalência de desnutrição e obesidade infantil nos municípios brasileiros: Alogaritmos de Machine Learnng
# 
# #### Audencio Victor, Ruchelli França de Lima e  Alexandre Chiavegatto Filho
# 
# ##### Dataset:Enani 
# 
# ##### Objetivo da Análise: Predizer a prevalência de desnutrição e obesidade infantil nos municípios brasileiros
# 
# ##### O Conjunto de dados(ENANI)possui 1455 observações, com as as seguintes variáveis listadas:

# ### O Conjunto de dados possui 14560(14560)  criancas de 120 municipios, com as as seguintes variáveis listadas:
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

# In[33]:


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

# In[347]:


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

# In[694]:


# Lista com os nomes das colunas que você deseja importar
var = ['id_desidentificado','a00_regiao','a01_uf','a02_municipio', 'a11_situacao','b05a_idade_em_meses','anemia',
                     'deficiencia_vita', 'vd_zwaz_categ','vd_zimc_categ','vd_zhaz_categ'] 


# In[695]:


# Lê apenas as colunas desejadas no Dataset 
df = pd.read_excel("dataset_enani_final.xlsx", usecols=var)
df.head()


# In[696]:


# criando uma Varivel populatoat por minicipio 
df['pop'] = df['a02_municipio']
df.columns


# ### Transformar as categorias dos desfechos em numero computavel - Para calcular a prevalencia ###

# In[697]:


#### Desnutricao  por peso/idade 
a =df['vd_zwaz_categ'].unique()
print (a)

# Definir o dicionário de mapeamento
mapeamento = {'Peso adequado': 0, 'Peso elevado': 0, 'Baixo peso': 1, 'Muito baixo peso': 1}

# Recodificar a coluna 'vd_zwaz_categ' do DataFrame 'df' usando o método map()
df['vd_zwaz_categ'] = df['vd_zwaz_categ'].map(mapeamento)

# Mostrar o resultado
print(df['vd_zwaz_categ'].unique())


# In[698]:


# Calcular o  total de crincas desnutridas por Peso/idade
somatorio_desnutricao = df['vd_zwaz_categ'].sum()

# Mostrar o resultado
print("Total de crincas desnutridas por Peso/idade:", somatorio_desnutricao)


# #### Desnutricao  por Estatura/idade

# In[699]:


#### Desnutricao  por Estatura/idade 
b =df['vd_zhaz_categ'].unique()
print (b)

# Definir o dicionário de mapeamento
mapeamento = {'Altura adequada': 0, 'Baixa altura': 1, 'Muito baixa altura': 1}

# Recodificar a coluna 'vd_zwaz_categ' do DataFrame 'df' usando o método map()
df['vd_zhaz_categ'] = df['vd_zhaz_categ'].map(mapeamento)

# Mostrar o resultado
print(df['vd_zhaz_categ'].unique())


# In[700]:


# Calcular o  total de crincas desnutridas por Estatura/idade
somatorio_desnutricaop = df['vd_zhaz_categ'].sum()

# Mostrar o resultado
print("Total de crincas desnutridas por Estatura/idade:", somatorio_desnutricaop)


# #### Obesidade  por IMC/idade 

# In[701]:


c = df['vd_zimc_categ'].unique()
print(c)

# Definir o dicionário de mapeamento
mapeamento = {'Eutrofia': 0,'Magreza': 0,'Magreza acentuada': 0,'Risco de sobrepeso': 0,'Obesidade': 1,
              'Sobrepeso': 0}

# Recodificar a coluna 'vd_zwaz_categ' do DataFrame 'df' usando o método map()
df['vd_zimc_categ'] = df['vd_zimc_categ'].map(mapeamento)

# Mostrar o resultado
print(df['vd_zimc_categ'].unique())


# In[702]:


## Calcular o  total de crincas obesas pelo IMC para idade 
somatorio_obesida = df['vd_zimc_categ'].sum()

# Mostrar o resultado
print("Total de obesas:", somatorio_obesida)


# #### Viatamina A

# In[703]:


# # Calcular o  total de crincas com defiencia de Vita A 
somatorio_vita = df['deficiencia_vita'].sum()

# Mostrar o resultado
print("Total de crincas com defiencia de Vita A :", somatorio_vita )


# #### Anemia

# In[704]:


# # Calcular o total de crincas anemicas 
somatorio_anemia = df['anemia'].sum()

# Mostrar o resultado
print("Total de crincas anemicas :", somatorio_anemia)


# #### Transformando a variavel idade em 2 grupos: ( O = abaixo de 24 meses e 1 = acima de 24 meses)

# In[705]:


import re

# Passo 1: Converter a coluna 'b05a_idade_em_meses' para strings e extrair apenas os números usando expressões regulares
df['b05a_idade_em_meses'] = df['b05a_idade_em_meses'].astype(str).str.extract(r'(\d+)').astype(float)

# Passo 2: Criar uma nova coluna 'idade_classificada' com valores 0 ou 1
df['b05a_idade_em_meses'] = df['b05a_idade_em_meses'].apply(lambda idade: 1 if idade >= 25 else 0)

# Exibir o DataFrame resultante
print(df)


# In[706]:


df.head()


# In[707]:


df.info() # tomando informações sobre o tipo das variáveis parametrizadas na chamada


# In[710]:


df.tail
df.columns  # para ver os nomes das  variveis 
list(df.columns.values.tolist()) # listar todas as variaveis 


# In[711]:


## Numero toral de municipios 
v = df['a02_municipio'].nunique()
print("Número de municípios:", v)


# #### Organizacao do banco e transformacao para o agregado 

# In[731]:


# Agregar novamente apenas por municípios para obter os totais para cada município

df_agre = df.groupby(['a02_municipio','a00_regiao','a01_uf']).agg({
    'anemia': 'sum',
    'deficiencia_vita': 'sum',
    'vd_zimc_categ': 'sum',
    'vd_zhaz_categ': 'sum',
    'vd_zwaz_categ': 'sum',
    'pop': 'count'  # Contagem total de ocorrências por município
}).reset_index()


# In[732]:


df_agre.head()


# In[733]:


df_agre.info()
## Numero toral de municipios 


# #### Calcular a prevalência dos desfechos  (Desnutricao, Obesidade, Viatamina A e Anemia)

# In[734]:


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


# In[735]:


print(df_agre)


# In[736]:


df_agre.info()


# ###### As analises desccritivas no Banco de dados 

# In[737]:


# Agrupar por região e calcular a média da prevalência de cada desfecho

prevalencia_regiao = df_agre.groupby('a00_regiao').mean()[['Prev_desnut', 'Prev_desnut1','Prev_Obesi', 'Prev_anem', 'Prev_viaa']]

print(prevalencia_regiao)


# In[738]:


# Calcular a média geral da prevalência de cada desfecho
prevalencia_geral = df_agre.mean()[['Prev_desnut', 'Prev_desnut1', 'Prev_Obesi', 'Prev_anem', 'Prev_viaa']]

print("Prevalência geral:")
print(prevalencia_geral)


# In[739]:


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


# In[740]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.boxplot([df_agre['anemia'], df_agre['deficiencia_vita'], df_agre['vd_zimc_categ'], df_agre['vd_zhaz_categ'], df_agre['vd_zwaz_categ']],
            labels=['Anemia', 'Deficiência de Vitamina A', 'Desnutrição ZIMC', 'Desnutrição ZHAZ', 'Desnutrição ZWAZ'])
plt.title('Boxplot das Prevalências dos Desfechos')
plt.ylabel('Prevalência')

plt.show()


# In[741]:



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

# In[742]:



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


# In[743]:


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

# In[745]:


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

# In[746]:


df_agre0 = df_agre1 [df_agre1['b05a_idade_em_meses']== 0]
df_agre0.head()


# In[747]:


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


# In[748]:


print(df_agre0)


# In[749]:


# Agrupar por região e calcular a média da prevalência de cada desfecho

prevalencia_regiao0 = df_agre0.groupby('a00_regiao').mean()[['Prev_desnut', 'Prev_desnut1','Prev_Obesi', 'Prev_anem', 'Prev_viaa']]

print(prevalencia_regiao0)


# In[753]:


# Calcular a média geral da prevalência de cada desfecho
prevalencia_geral0 = df_agre0.mean()[['Prev_desnut', 'Prev_desnut1', 'Prev_Obesi', 'Prev_anem', 'Prev_viaa']]

print("Prevalência geral0:")
print(prevalencia_geral0)


# In[754]:


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

# In[755]:


df_agre1 = df_agre1[df_agre1['b05a_idade_em_meses'] == 1]
df_agre1.head()


# In[756]:


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


# In[757]:


print(df_agre1)


# In[778]:


# Agrupar por região e calcular a média da prevalência de cada desfecho

prevalencia_regiao1 = df_agre1.groupby('a00_regiao').mean()[['Prev_desnut', 'Prev_desnut1','Prev_Obesi', 'Prev_anem', 'Prev_viaa']]

print(prevalencia_regiao1)


# In[781]:


# Calcular a média geral da prevalência de cada desfecho
prevalencia_geral1 = df_agre1.mean()[['Prev_desnut', 'Prev_desnut1', 'Prev_Obesi', 'Prev_anem', 'Prev_viaa']]

print("Prevalência geral1:")
print(prevalencia_geral1)


# In[760]:



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

# In[1446]:


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

# In[1776]:


# Convert 'a02_municipio' column to string in both DataFrames
dfn['a02_municipio'] = dfn['a02_municipio'].astype(str)
df_agre['a02_municipio'] = df_agre['a02_municipio'].astype(str)

# Now, perform the merge operation
df_merge = dfn.merge(df_agre, on='a02_municipio', how='left')


# In[1777]:


# Merge usando o campo 'id' como chave
df_merge = dfn.merge(df_agre, on='a02_municipio', how='left')
df_merge.dropna(subset=['Prev_desnut', 'Prev_desnut1', 'Prev_Obesi', 'Prev_anem', 'Prev_viaa'], inplace=True)


# In[1778]:


df_merge.head()


# In[1779]:


df_merge.info() 


# In[1780]:


#Covertentendo a variaval Populacao  e Regiao  object  de para Flot (quantitativa)
df_merge['População'] = df_merge['População'].str.replace('.', '').astype(float)


# In[1781]:


pd.set_option('display.max_columns', None)
# Print the DataFrame
print(df_merge)


# In[1782]:


# Agrupar por região e calcular a média da prevalência de cada desfecho

prevalencia_regiao = df_merge.groupby('a00_regiao').mean()[['Prev_desnut', 'Prev_desnut1','Prev_Obesi', 'Prev_anem', 'Prev_viaa']]

print(prevalencia_regiao)


# In[1783]:


# Agrupar por região e calcular a média da prevalência de cada desfecho

prevalencia_estado = df_merge.groupby(['UF','a00_regiao']).mean()[['Prev_desnut', 'Prev_desnut1','Prev_Obesi', 'Prev_anem', 'Prev_viaa']]
prevalencia_estado.sort_values(by='a00_regiao', ascending=True, inplace=True)
print(prevalencia_estado)


# In[1784]:



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


# In[1796]:


# Mudando o nomde do dataset 
df = df_merge
df.head()


# In[1797]:


df.isna()
df.isna().sum()


# In[1798]:


df.drop(['deficiencia_vita', 'vd_zimc_categ','anemia', 'vd_zhaz_categ', 'vd_zwaz_categ','pop'], axis=1, inplace=True)


# In[1799]:


# Calcular a matriz de correlação
correlation_matrix = df.corr()

print(correlation_matrix)


# In[1800]:


# Calcular a matriz de correlação
correlation_matrix = df.corr()['Prev_desnut']

print(correlation_matrix)


# In[1801]:


# Calcular a matriz de correlação com o desfecho
correlation_matrix = df.corr()['Prev_desnut'].to_frame()

# Criar o mapa de calor
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Exibir o gráfico
plt.title('Correlação com o Desfecho')
plt.show()


# In[ ]:





# In[1802]:



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


# In[1803]:


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


# In[1804]:


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


# In[1805]:


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


# In[1806]:


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


# In[1763]:


df.columns


# In[1764]:


#Apagando variaveis do dataset 
df.drop(['a02_municipio','População', 'UF', 'pop_rural','PortMunicip', 'pop_urbana','Municip', 'anemia','desemprego',
      'analfa_total', 'desemprego', 'analfa_total','analfa_mulher','a00_regiao','pibpcapita','idhm_longevidade',
       'deficiencia_vita', 'vd_zimc_categ', 'vd_zhaz_categ', 'vd_zwaz_categ', 'CoberAB1','renda_gini','CoberESF1',
       'pop', 'Prev_desnut1', 'Prev_Obesi', 'Prev_anem','a01_uf','idhm_renda','RendDomiciliar',
       'Prev_viaa'], axis=1, inplace=True)


# In[1765]:


# Para as variáveis categóricas iremos criar dummies/ One hotencode
#df= pd.get_dummies(df, columns=['a00_regiao'])


# In[1766]:


##imputar dados pela media dsa variavel (Separar entre treino e teste)
df.fillna(df.mean(), inplace=True)


# In[1767]:


# variável de interesse/desfecho 
target= df >> select(X.Prev_desnut)


# In[1768]:


#Defenindo as variveis preditoras

variaveis_preditoras = df.iloc[:, df.columns != 'Prev_desnut']
classe = df.iloc[:, df.columns == 'Prev_desnut']
X_train, X_test, y_train, y_test = train_test_split(variaveis_preditoras, 
                                                    classe,
                                                    train_size = 0.70,
                                                    random_state = 45)


# In[1769]:


X_train.shape


# In[1770]:


X_test.shape


# In[1771]:


X_test.columns


# In[1772]:


X_train.columns


# In[1773]:


# Standarscaler com passthrough tem um problema de ordenação das colunas. Quando aplicamos, ele fornce o resultado com as colunas padronizadas em primeiro, seguidas das demais colunas.
# Para resolver este problema, iremos ordenar as nossas colunas alocando as contínuas nas primeiras posições 

X_train = X_train.loc[:,[ 'RendDomiciliar', 'Densestasaudaveis', 'DensestaNaosaudaveis', 'idhm',
       'idhm_renda', 'idhm_longevidade', 'idhm_educacao', 'pibpcapita',
       'renda', 'renda_gini', 'esc_anos', 'pobres', 'extr_pobres', 'CoberESF1',
       'a01_uf']]

X_test = X_test.loc[:,[ 'RendDomiciliar', 'Densestasaudaveis', 'DensestaNaosaudaveis', 'idhm',
       'idhm_renda', 'idhm_longevidade', 'idhm_educacao', 'pibpcapita',
       'renda', 'renda_gini', 'esc_anos', 'pobres', 'extr_pobres', 'CoberESF1',
       'a01_uf']]

X_train_columns = X_train.columns
X_test_columns = X_test.columns


# In[1774]:


from sklearn.compose import ColumnTransformer

### variáveis contínuas que serão padronizadas
continuous_cols = ['RendDomiciliar', 'Densestasaudaveis', 'DensestaNaosaudaveis', 'idhm',
       'idhm_renda', 'idhm_longevidade', 'idhm_educacao', 'pibpcapita',
       'renda', 'renda_gini', 'esc_anos', 'pobres', 'extr_pobres', 'CoberESF1',
       'a01_uf']

def setScaler():
  ct = ColumnTransformer([
        ('scaler', StandardScaler(), continuous_cols)
    ], remainder='passthrough' # utilizamos para manter as colunas em que não aplicamos o scaler
  )
  return ct
  
scaler = setScaler()


# In[1430]:


scaler.fit(X_train)


# In[1431]:


X_train = scaler.transform(X_train)


# In[1432]:


X_test = scaler.transform(X_test)


# In[1433]:


# para evitarmos a exibição dos dados em notacao científica
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# In[1434]:


X_train = pd.DataFrame(X_train, columns=X_train_columns)
X_test = pd.DataFrame(X_test, columns=X_test_columns)


# In[1435]:


X_train.head()


# In[1436]:


X_test.describe()


# In[1437]:


get_ipython().system('pip install pycaret')
from pycaret.regression import *

# configurar experimento do PyCaret
clf = setup(df, target='Prev_desnut')

# comparar modelos
best_model = compare_models()


# In[ ]:


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


# In[ ]:


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


# In[1443]:


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


# In[1444]:


print(score)


# In[1530]:


import pyforest
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


# In[1775]:


reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Print the results
print(models)


# In[ ]:





# In[ ]:





# In[ ]:




