import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

# função que calcula as estatisticas para cada coluna
def estatisticas(coluna):
    return {
        "Média": coluna.mean(),
        "Mediana": coluna.median(),
        "Moda": coluna.mode().iloc[0],
        "Mínimo": coluna.min(),
        "Máximo": coluna.max(),
        "Q1": coluna.quantile(0.25),
        "Q3": coluna.quantile(0.75),
        "Variância": coluna.var(),
        "Desvio padrão": coluna.std(),
        "Coeficiente de variação (%)": (coluna.std() / coluna.mean()) * 100
    }

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_excel(BASE_DIR / "Online Retail.xlsx")

# print(df.info())
#transformações de dados

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']) #converte para tipo de data

df['TotalPrice'] = df['Quantity'] * df['UnitPrice'] #transforma preço unitário em preço total

df["InvoiceNo"] = df["InvoiceNo"].astype(str) #converte a coluna para string
df = df[~df['InvoiceNo'].str.startswith('C')] #remove compras canceladas

df = df.dropna(subset=['CustomerID']) #remove clientes sem ID


#agrupa clientes 
clientes = (
    df.groupby("CustomerID")
      .agg(
          Frequencia=("InvoiceNo", "nunique"),
          ValorTotal=("TotalPrice", "sum"),
          QuantidadeTotal=("Quantity", "sum"),
          ProdutosDistintos=("StockCode", "nunique")
      )
      .reset_index()
)

#cria mais duas variáveis quantitativas além das 4 agrupadas antes
clientes["ValorMedioPorCompra"] = (
    clientes["ValorTotal"] / clientes["Frequencia"]
)

clientes["QuantidadeMediaPorCompra"] = (
    clientes["QuantidadeTotal"] / clientes["Frequencia"]
)

#chama função q calcula estatísticas das variáveis
freq = estatisticas(clientes["Frequencia"])
valorTotal = estatisticas(clientes["ValorTotal"])
quantidadeTotal = estatisticas(clientes["QuantidadeTotal"])
produtos = estatisticas(clientes["ProdutosDistintos"])
valorMedio = estatisticas(clientes["ValorMedioPorCompra"])
quantidadeMedia = estatisticas(clientes["QuantidadeMediaPorCompra"])

#vetor para guardar nomes das colunas
variaveis = [
    "Frequencia",
    "ValorTotal",
    "QuantidadeTotal",
    "ProdutosDistintos",
    "ValorMedioPorCompra",
    "QuantidadeMediaPorCompra"
]

# for que faz um boxplot pra cada coluna
for variavel in variaveis:
    plt.figure(figsize=(6,4))
    plt.boxplot(clientes[variavel])
    plt.title(f"Boxplot - {variavel}")
    plt.ylabel(variavel)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 4.4) 
# separa as colunas quantitativas para a analise
colunas_analise = [
    "Frequencia", "ValorTotal", "QuantidadeTotal", 
    "ProdutosDistintos", "ValorMedioPorCompra", "QuantidadeMediaPorCompra"
]
df_multivariado = clientes[colunas_analise]

# calcula a matriz de correlacao entre as variaveis
matriz_corr = df_multivariado.corr()

# cria o mapa de calor das correlacoes
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Mapa de Calor das Correlações")
plt.tight_layout()
plt.show()

# cria um grafico de dispersao entre frequencia e valor total
plt.figure(figsize=(6, 4))
plt.scatter(clientes["Frequencia"], clientes["ValorTotal"], alpha=0.5, color='purple')
plt.title("Dispersão: Frequência vs Valor Total")
plt.xlabel("Frequência")
plt.ylabel("Valor Total")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# 4.5) 
# mantem as 6 variaveis ja criadas para o agrupamento
variaveis_selecionadas = colunas_analise
X_original = clientes[variaveis_selecionadas]

# 4.6)
# aplica a padronizacao z-score nas variaveis
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_original)

# 6.1) 
wcss = []          # vetor para guardar o wcss (cotovelo)
silhuetas = []     # vetor para guardar o coeficiente de silhueta
K_range = range(2, 11) # testa grupos de 2 a 10

# for que roda o kmeans para cada valor de K
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    rotulos = kmeans.fit_predict(X_scaled)
    
    wcss.append(kmeans.inertia_)
    
    score_silhueta = silhouette_score(X_scaled, rotulos, metric='euclidean')
    silhuetas.append(score_silhueta)

# grafico do metodo do cotovelo
plt.figure(figsize=(6, 4))
plt.plot(K_range, wcss, marker='o', linestyle='--', color='blue')
plt.title("Método do Cotovelo")
plt.xlabel("Número de Grupos (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# grafico do coeficiente de silhueta
plt.figure(figsize=(6, 4))
plt.plot(K_range, silhuetas, marker='s', linestyle='-', color='green')
plt.title("Coeficiente de Silhueta")
plt.xlabel("Número de Grupos (K)")
plt.ylabel("Silhueta Média")
plt.grid(True)
plt.show()


# MODELO FINAL E PERFIL DOS GRUPOS
# roda o kmeans final com 3 grupos
k_escolhido = 3 
kmeans_final = KMeans(n_clusters=k_escolhido, random_state=42, n_init=20)
clientes['Grupo_KMeans'] = kmeans_final.fit_predict(X_scaled)

# conta quantos clientes ficaram em cada grupo
print("\nTamanho dos Grupos:")
print(clientes['Grupo_KMeans'].value_counts())

# calcula a media original das variaveis para cada grupo
perfil_grupos = clientes.groupby('Grupo_KMeans')[variaveis_selecionadas].mean()
print("\nPerfil Médio de Cada Grupo:")
print(perfil_grupos)

import scipy.cluster.hierarchy as sch
from sklearn.cluster import DBSCAN

# 6.2) 
# O algoritmo hierarquico consome muita memoria, entao usamos uma amostra
X_amostra = X_scaled[:1000] 

# gera o dendrograma usando o metodo de Ward e distancia euclidiana
plt.figure(figsize=(10, 6))
dendrograma = sch.dendrogram(sch.linkage(X_amostra, method='ward', metric='euclidean'))
plt.title("Dendrograma - Agrupamento Hierárquico")
plt.xlabel("Clientes (Amostra)")
plt.ylabel("Distância")
plt.show()

# 6.3) 
# configura o dbscan para encontrar grupos por densidade
dbscan = DBSCAN(eps=0.5, min_samples=5)
clientes['Grupo_DBSCAN'] = dbscan.fit_predict(X_scaled)

# conta quantos clientes ficaram em cada grupo do dbscan (-1 sao os ruidos)
print("\nTamanho dos Grupos no DBSCAN:")
print(clientes['Grupo_DBSCAN'].value_counts())

# salva os dados que serao utilizados na segunda parte
dados_para_proxima_etapa = {
    "clientes": clientes,
    "X_scaled": X_scaled,
    "X_amostra": X_amostra,
    "variaveis_selecionadas": variaveis_selecionadas,
    "kmeans_final": kmeans_final,
}

with open("dados_etapa1.pkl", "wb") as arquivo:
    pickle.dump(dados_para_proxima_etapa, arquivo)

print("\nDados salvos em 'dados_etapa1.pkl' para uso no SegundaParteTrabalhoFinal.py")