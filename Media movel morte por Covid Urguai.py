#!/usr/bin/env python
# coding: utf-8

# In[49]:


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


# In[50]:


# Lê apenas as colunas desejadas no Dataset 
df = pd.read_excel("UR 2021.xlsx")
df.head(20)


# In[51]:


window_size = 7
df['new_cases_ma'] = df['new_cases_smoothed_per_million'].rolling(window=window_size, min_periods=1).mean()
df['new_deaths_ma'] = df['new_deaths_smoothed_per_million'].rolling(window=window_size, min_periods=1).mean()


# In[52]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Plotting the original data
plt.plot(df['date'], df['new_cases_smoothed_per_million'], label='New Cases per Million')
plt.plot(df['date'], df['new_deaths_smoothed_per_million'], label='New Deaths per Million')

# Plotting the moving averages
plt.plot(df['date'], df['new_cases_ma'], label='New Cases Moving Avg.')
plt.plot(df['date'], df['new_deaths_ma'], label='New Deaths Moving Avg.')

plt.xlabel('Date')
plt.ylabel('Cases / Million')
plt.title('Moving Averages of New Cases and Deaths per Million')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()


# In[53]:


import pandas as pd
import matplotlib.pyplot as plt

# Convertendo a coluna 'date' para o tipo de data
df['date'] = pd.to_datetime(df['date'])

# Ordenando os dados pela coluna 'date'
df = df.sort_values(by='date')

# Calculando a média móvel de 7 dias para a coluna 'new_deaths_smoothed_per_million'
df['new_deaths_smoothed_per_million_7d'] = df['new_deaths_smoothed_per_million'].rolling(window=7).mean()
df['new_cases_smoothed_per_million_7d'] = df['new_cases_smoothed_per_million'].rolling(window=7).mean()

# Filtrando apenas as colunas relevantes para o gráfico
df_deaths_filtered = df[['date', 'new_deaths_smoothed_per_million_7d']]
df_cases_filtered = df[['date', 'new_cases_smoothed_per_million_7d']]

# Removendo as linhas com valores NaN resultantes do cálculo da média móvel
df_deaths_filtered = df_deaths_filtered.dropna()
df_cases_filtered = df_cases_filtered.dropna()

# Criando o gráfico de médias móveis de 7 dias de novos óbitos por milhão de habitantes e novos casos por milhão de habitantes
plt.figure(figsize=(12, 6))
plt.plot(df_deaths_filtered['date'], df_deaths_filtered['new_deaths_smoothed_per_million_7d'], label='Média Móvel de 7 dias de Novos Óbitos')
plt.plot(df_cases_filtered['date'], df_cases_filtered['new_cases_smoothed_per_million_7d'], label='Média Móvel de 7 dias de Novos Casos')

plt.xlabel('Data')
plt.ylabel('Média Móvel por Milhão')
plt.title('Médias Móveis de 7 dias de Novos Casos e Novas Mortes por Milhão')
# Definindo intervalo quinzenal no eixo x
plt.xticks(df['date'][::15], rotation=45)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[54]:


# Plot dos dados originais
plt.plot(df['date'], df['new_cases_smoothed_per_million'], label='Novos Casos por Milhão')
plt.plot(df['date'], df['new_deaths_smoothed_per_million'], label='Novas Mortes por Milhão')

# Plot das médias móveis
plt.plot(df['date'], df['new_cases_ma'], label='Média Móvel de Novos Casos')
plt.plot(df['date'], df['new_deaths_ma'], label='Média Móvel de Novas Mortes')

plt.xlabel('Data')
plt.ylabel('Casos / Milhão')
plt.title('Média Móvel de 7 dias de Novos Casos e Mortes por Milhão')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Mostrar o gráfico
plt.show()


# In[48]:


# Convertendo a coluna 'date' para o tipo de data
df['date'] = pd.to_datetime(df['date'])

# Ordenando os dados pela coluna 'date'
df = df.sort_values(by='date')

# Calculando a média móvel de 7 dias para a coluna 'new_deaths_smoothed_per_million'
df['new_deaths_smoothed_per_million_7d'] = df['new_deaths_smoothed_per_million'].rolling(window=7).mean()

# Filtrando apenas as colunas relevantes para o gráfico
df_filtered = df[['date', 'new_deaths_smoothed_per_million_7d']]

# Removendo as linhas com valores NaN resultantes do cálculo da média móvel
df_filtered = df_filtered.dropna()

# Criando o gráfico de médias móveis de 7 dias de novos óbitos por milhão de habitantes
plt.figure(figsize=(12, 6))
plt.plot(df_filtered['date'], df_filtered['new_deaths_smoothed_per_million_7d'], label='Média Móvel de 7 dias')
plt.xlabel('Data')
plt.ylabel('Novos óbitos por milhão de habitantes')
plt.title('Média Móvel de 7 dias de Novos Óbitos por Milhão de Habitantes')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[45]:



# Criando o gráfico de médias móveis de 7 dias de novos óbitos por milhão de habitantes
plt.figure(figsize=(12, 6))
plt.plot(df_filtered['date'], df_filtered['new_deaths_smoothed_per_million_7d'], label='Média Móvel de 7 dias')

plt.xlabel('Data')
plt.ylabel('Novos óbitos por milhão de habitantes')
plt.title('Média Móvel de 7 dias de Novos Óbitos por Milhão de Habitantes')
plt.legend()
plt.grid(True)

# Definindo intervalo quinzenal no eixo x
plt.xticks(df_filtered['date'][::15], rotation=45)

plt.tight_layout()
plt.show()


# In[ ]:




