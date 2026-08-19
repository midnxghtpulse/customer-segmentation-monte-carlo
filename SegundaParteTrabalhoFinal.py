"""
essa é a segunda parte do trabalho. este arquivo deve ser executado APÓS o PrimeiraParteTrabalhoFinal.py. como cada script roda de forma independente, 
os dados necessários são carregados aqui a partir do arquivo 'dados_etapa1.pkl', que é gerado automaticamente no final do PrimeiraParteTrabalhoFinal.py. 
ele cobre os pontos 7, 8 e 9 do trabalho.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)

# carrega os dados salvos da primwira parte

try:
    with open("dados_etapa1.pkl", "rb") as arquivo:
        dados_etapa1 = pickle.load(arquivo)
except FileNotFoundError:
    raise FileNotFoundError(
        "Arquivo 'dados_etapa1.pkl' não encontrado. "
        "Execute primeiro o PrimeiraParteTrabalhoFinal.py "
        "(ele deve estar na mesma pasta e gera esse arquivo automaticamente)."
    )

clientes = dados_etapa1["clientes"]
X_scaled = dados_etapa1["X_scaled"]
X_amostra = dados_etapa1["X_amostra"]
variaveis_selecionadas = dados_etapa1["variaveis_selecionadas"]
kmeans_final = dados_etapa1["kmeans_final"]

print("Arquuivo rodou")

# parte 6.2

k_hier = 3 

hier = AgglomerativeClustering(n_clusters=k_hier, linkage="ward")
rotulos_hier_amostra = hier.fit_predict(X_amostra)

# pwrte 6.3 (k-distance plot)

min_samples_dbscan = 5

vizinhos = NearestNeighbors(n_neighbors=min_samples_dbscan)
vizinhos_fit = vizinhos.fit(X_scaled)
distancias, _ = vizinhos_fit.kneighbors(X_scaled)

# ordena as distâncias do k-ésimo vizinho mais próximo

k_distancias = np.sort(distancias[:, -1])

plt.figure(figsize=(6, 4))
plt.plot(k_distancias)
plt.title("Gráfico das distâncias dos vizinhos (k-distance)")
plt.xlabel("Pontos ordenados")
plt.ylabel(f"Distância ao {min_samples_dbscan}º vizinho mais próximo")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# testando alguns valores de eps e min_samples para justificar a escolha

eps_testados = [0.3, 0.5, 0.7, 1.0]
min_samples_testados = [5, 10, 15]

resultados_dbscan_testes = []
for eps in eps_testados:
    for ms in min_samples_testados:
        modelo = DBSCAN(eps=eps, min_samples=ms)
        rot = modelo.fit_predict(X_scaled)
        n_grupos = len(set(rot)) - (1 if -1 in rot else 0)
        n_ruido = int(np.sum(rot == -1))
        resultados_dbscan_testes.append(
            {"eps": eps, "min_samples": ms, "n_grupos": n_grupos, "n_ruido": n_ruido}
        )

tabela_dbscan_testes = pd.DataFrame(resultados_dbscan_testes)
print("\nTeste de parâmetros do DBSCAN:")
print(tabela_dbscan_testes)

# avaliacao e comparacao entre metodos

rot_kmeans = clientes["Grupo_KMeans"].values
sil_kmeans = silhouette_score(X_scaled, rot_kmeans)
db_kmeans = davies_bouldin_score(X_scaled, rot_kmeans)
n_grupos_kmeans = len(set(rot_kmeans))
n_ruido_kmeans = 0

# hierarquico
sil_hier = silhouette_score(X_amostra, rotulos_hier_amostra)
db_hier = davies_bouldin_score(X_amostra, rotulos_hier_amostra)
n_grupos_hier = len(set(rotulos_hier_amostra))
n_ruido_hier = 0

# dbscan
rot_dbscan = clientes["Grupo_DBSCAN"].values
mask_validos = rot_dbscan != -1 
if len(set(rot_dbscan[mask_validos])) > 1:
    sil_dbscan = silhouette_score(X_scaled[mask_validos], rot_dbscan[mask_validos])
    db_dbscan = davies_bouldin_score(X_scaled[mask_validos], rot_dbscan[mask_validos])
else:
    sil_dbscan, db_dbscan = np.nan, np.nan
n_grupos_dbscan = len(set(rot_dbscan)) - (1 if -1 in rot_dbscan else 0)
n_ruido_dbscan = int(np.sum(rot_dbscan == -1))

tabela_comparacao = pd.DataFrame(
    {
        "Método": ["K-means", "Hierárquico (amostra)", "DBSCAN"],
        "Grupos": [n_grupos_kmeans, n_grupos_hier, n_grupos_dbscan],
        "Silhueta": [sil_kmeans, sil_hier, sil_dbscan],
        "Davies-Bouldin": [db_kmeans, db_hier, db_dbscan],
        "Ruído (n)": [n_ruido_kmeans, n_ruido_hier, n_ruido_dbscan],
        "Ruído (%)": [
            0.0,
            0.0,
            100 * n_ruido_dbscan / len(clientes),
        ],
    }
)

print("\n" + "=" * 60)
print("Tabela de comparação")
print("=" * 60)
print(tabela_comparacao.to_string(index=False))

# k-means
silhuetas_individuais_kmeans = None
from sklearn.metrics import silhouette_samples

silhuetas_individuais_kmeans = silhouette_samples(X_scaled, rot_kmeans)
clientes["Silhueta_KMeans"] = silhuetas_individuais_kmeans

plt.figure(figsize=(6, 4))
sns.boxplot(x=clientes["Grupo_KMeans"], y=clientes["Silhueta_KMeans"])
plt.title("Distribuição da silhueta por grupo (K-means)")
plt.xlabel("Grupo")
plt.ylabel("Coeficiente de silhueta")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

n_silhuetas_negativas = int(np.sum(silhuetas_individuais_kmeans < 0))
print(f"\nObservações com silhueta negativa (K-means): {n_silhuetas_negativas} "
      f"({100 * n_silhuetas_negativas / len(clientes):.2f}%)")

# simulacao de monte carlo

scaler_ref = StandardScaler()
X_ref = scaler_ref.fit_transform(clientes[variaveis_selecionadas])

k_referencia = 3
modelo_ref = KMeans(n_clusters=k_referencia, random_state=42, n_init=20)
rotulos_ref = modelo_ref.fit_predict(X_ref)

dados_ref = clientes[variaveis_selecionadas].copy()
dados_ref.index = clientes.index

# reamostragem
B = 200
valores_ari = []
valores_silhueta = []
valores_db = []

for b in range(B):
    amostra = dados_ref.sample(frac=0.80, replace=False, random_state=b)
    indices = amostra.index

    scaler_b = StandardScaler()
    X_b = scaler_b.fit_transform(amostra)

    modelo_b = KMeans(n_clusters=k_referencia, random_state=b, n_init=20)
    rotulos_b = modelo_b.fit_predict(X_b)

    posicoes = dados_ref.index.get_indexer(indices)
    rotulos_ref_amostra = rotulos_ref[posicoes]

    ari = adjusted_rand_score(rotulos_ref_amostra, rotulos_b)

    if len(set(rotulos_b)) > 1:
        silhueta_b = silhouette_score(X_b, rotulos_b)
        db_b = davies_bouldin_score(X_b, rotulos_b)
    else:
        silhueta_b, db_b = np.nan, np.nan

    valores_ari.append(ari)
    valores_silhueta.append(silhueta_b)
    valores_db.append(db_b)

valores_ari = np.array(valores_ari)
valores_silhueta = np.array(valores_silhueta)
valores_db = np.array(valores_db)

# reamostragem 8.7
print("\n" + "=" * 60)
print("Resultados da Monte Carlo (com 80% de reamostragem)")
print("=" * 60)
print(f"ARI médio:      {np.mean(valores_ari):.4f}")
print(f"ARI mediana:    {np.median(valores_ari):.4f}")
print(f"ARI desvio-pad: {np.std(valores_ari):.4f}")
print(f"ARI mínimo:     {np.min(valores_ari):.4f}")
print(f"ARI máximo:     {np.max(valores_ari):.4f}")
ic95_ari = np.percentile(valores_ari, [2.5, 97.5])
print(f"IC 95% (percentílico): [{ic95_ari[0]:.4f}, {ic95_ari[1]:.4f}]")

limiar_instabilidade = 0.5
freq_instaveis = np.mean(valores_ari < limiar_instabilidade) * 100
print(f"Frequência de resultados instáveis (ARI < {limiar_instabilidade}): "
      f"{freq_instaveis:.2f}%")

# histograma do ARI
plt.figure(figsize=(6, 4))
plt.hist(valores_ari, bins=20, color="steelblue", edgecolor="black", alpha=0.8)
plt.axvline(np.mean(valores_ari), color="red", linestyle="--", label="Média")
plt.title("Histograma do ARI (reamostragem")
plt.xlabel("ARI")
plt.ylabel("Frequência")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
plt.figure(figsize=(6, 4))
plt.boxplot(valores_silhueta[~np.isnan(valores_silhueta)])
plt.title("Boxplot da silhueta nas simulações de Monte Carlo")
plt.ylabel("Coeficiente de silhueta")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

# indicice davis-boulding
plt.figure(figsize=(6, 4))
plt.hist(valores_db[~np.isnan(valores_db)], bins=20, color="darkorange",
         edgecolor="black", alpha=0.8)
plt.title("Distribuição do índice Davies-Bouldin nas simulações")
plt.xlabel("Davies-Bouldin")
plt.ylabel("Frequência")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# 8.5 com seed aleatoria

sementes = range(0, 30)
ari_sementes = []
centroides_sementes = []

for semente in sementes:
    modelo_s = KMeans(n_clusters=k_referencia, random_state=semente, n_init=1)
    rot_s = modelo_s.fit_predict(X_ref)
    ari_s = adjusted_rand_score(rotulos_ref, rot_s)
    ari_sementes.append(ari_s)
    centroides_sementes.append(modelo_s.cluster_centers_)

print("\n" + "=" * 60)
print("Estabilidade diante de diversas inicializações (seeds)")
print("=" * 60)
print(f"ARI médio entre sementes: {np.mean(ari_sementes):.4f}")
print(f"ARI desvio-padrão:        {np.std(ari_sementes):.4f}")
print(f"ARI mínimo:                {np.min(ari_sementes):.4f}")

plt.figure(figsize=(6, 4))
plt.plot(list(sementes), ari_sementes, marker="o", linestyle="--", color="teal")
plt.axhline(np.mean(ari_sementes), color="red", linestyle=":", label="Média")
plt.title("ARI por seed aleatória (inicialização do K-means)")
plt.xlabel("Seed (random_state)")
plt.ylabel("ARI em relação ao agrupamento de referência")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# 8.6

c_perturbacao = 0.05
desvios_originais = dados_ref.std()

ari_perturbacao = []
for b in range(B):
    rng = np.random.RandomState(b)
    ruido = rng.normal(
        loc=0.0,
        scale=desvios_originais.values * c_perturbacao,
        size=dados_ref.shape,
    )
    dados_perturbados = dados_ref.values + ruido

    scaler_p = StandardScaler()
    X_p = scaler_p.fit_transform(dados_perturbados)

    modelo_p = KMeans(n_clusters=k_referencia, random_state=42, n_init=20)
    rot_p = modelo_p.fit_predict(X_p)

    ari_p = adjusted_rand_score(rotulos_ref, rot_p)
    ari_perturbacao.append(ari_p)

ari_perturbacao = np.array(ari_perturbacao)

print("\n" + "=" * 60)
print(f"Estabilidade diante de perturbacões (c = {c_perturbacao})")
print("=" * 60)
print(f"ARI médio:   {np.mean(ari_perturbacao):.4f}")
print(f"ARI mínimo:  {np.min(ari_perturbacao):.4f}")
print(f"ARI máximo:  {np.max(ari_perturbacao):.4f}")

plt.figure(figsize=(6, 4))
plt.hist(ari_perturbacao, bins=20, color="mediumseagreen", edgecolor="black",
         alpha=0.8)
plt.title("Histograma do ARI - Perturbação das Observações")
plt.xlabel("ARI")
plt.ylabel("Frequência")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# 8.8 (o calculo é O(n^2))

n_coagrupamento = min(300, len(clientes))
indices_coagrupamento = clientes.sample(
    n=n_coagrupamento, random_state=42
).index

contagem_juntos = pd.DataFrame(
    0, index=indices_coagrupamento, columns=indices_coagrupamento, dtype=float
)
contagem_amostrado = pd.DataFrame(
    0, index=indices_coagrupamento, columns=indices_coagrupamento, dtype=float
)

for b in range(B):
    amostra = dados_ref.sample(frac=0.80, replace=False, random_state=b)
    presentes = amostra.index.intersection(indices_coagrupamento)
    if len(presentes) < 2:
        continue

    scaler_b = StandardScaler()
    X_b = scaler_b.fit_transform(amostra)
    modelo_b = KMeans(n_clusters=k_referencia, random_state=b, n_init=20)
    rot_b = modelo_b.fit_predict(X_b)

    rotulos_amostra_b = pd.Series(rot_b, index=amostra.index)
    sub = rotulos_amostra_b.loc[presentes]

    contagem_amostrado.loc[presentes, presentes] += 1
    mesmo_grupo = np.equal.outer(sub.values, sub.values).astype(float)
    contagem_juntos.loc[presentes, presentes] += mesmo_grupo

with np.errstate(divide="ignore", invalid="ignore"):
    matriz_coagrupamento = contagem_juntos / contagem_amostrado
matriz_coagrupamento = matriz_coagrupamento.fillna(0)

plt.figure(figsize=(8, 6))
sns.heatmap(matriz_coagrupamento, cmap="viridis", xticklabels=False,
            yticklabels=False)
plt.title(f"Matriz de Coagrupamento (amostra de {n_coagrupamento} clientes)")
plt.tight_layout()
plt.show()


# interpretacao dos grupos

print("\n" + "=" * 60)
print("CARACTERIZAÇÃO DETALHADA DOS GRUPOS (K-MEANS)")
print("=" * 60)

for variavel in variaveis_selecionadas:
    resumo = clientes.groupby("Grupo_KMeans")[variavel].agg(
        ["mean", "median", "std", "min", "max"]
    )
    print(f"\nVariável: {variavel}")
    print(resumo)

# grafico de dispersao
sns.pairplot(
    clientes,
    vars=variaveis_selecionadas,
    hue="Grupo_KMeans",
    palette="Set1",
    diag_kind="kde",
    plot_kws={"alpha": 0.5, "s": 20},
)
plt.suptitle("Grupos do K-means em Pares de Variáveis", y=1.02)
plt.show()

# ruido dbscan
outliers_dbscan = clientes[clientes["Grupo_DBSCAN"] == -1]
print(f"\nObservações classificadas como ruído pelo DBSCAN: {len(outliers_dbscan)} "
      f"({100 * len(outliers_dbscan) / len(clientes):.2f}%)")
print("\nEstatísticas descritivas dos clientes atípicos:")
print(outliers_dbscan[variaveis_selecionadas].describe())

print("\nEstatísticas descritivas dos clientes não atípicos (comparativo):")
print(clientes[clientes["Grupo_DBSCAN"] != -1][variaveis_selecionadas].describe())

centroides = kmeans_final.cluster_centers_
distancias_centroide = np.linalg.norm(
    X_scaled - centroides[clientes["Grupo_KMeans"].values], axis=1
)
clientes["DistanciaCentroide"] = distancias_centroide

limite_atipico = clientes["DistanciaCentroide"].quantile(0.99)
atipicos_kmeans = clientes[clientes["DistanciaCentroide"] > limite_atipico]
print(f"\nClientes mais distantes do centróide do seu grupo (top 1%): "
      f"{len(atipicos_kmeans)} observações")
print(atipicos_kmeans[["CustomerID", "Grupo_KMeans", "DistanciaCentroide"] +
                       variaveis_selecionadas].sort_values(
    "DistanciaCentroide", ascending=False
).head(10))

# exportacao da simulacao para entrwgar
resultados_simulacao = pd.DataFrame(
    {
        "repeticao": range(B),
        "ari_reamostragem": valores_ari,
        "silhueta_reamostragem": valores_silhueta,
        "davies_bouldin_reamostragem": valores_db,
        "ari_perturbacao": ari_perturbacao,
    }
)
resultados_simulacao.to_csv("resultados_monte_carlo.csv", index=False)

tabela_comparacao.to_csv("tabela_comparacao_metodos.csv", index=False)

print("\nArquivos exportados: resultados_monte_carlo.csv, "
      "tabela_comparacao_metodos.csv")