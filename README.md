# customer segmentation with clustering and monte carlo simulation

> note: this is an academic project from a couple of months ago. it was one of my first attempts at combining clustering techniques with monte carlo simulation, so i'm keeping it here as part of my learning process.

academic data analysis project focused on customer segmentation using unsupervised learning techniques and stability analysis.

## overview

the project uses transactional retail data to create customer-level features and compare different clustering approaches. it also evaluates the stability of the k-means solution through monte carlo resampling, different random initializations and perturbations in the observations.

## methods used

- exploratory data analysis
- descriptive statistics
- feature engineering
- data standardization with z-score
- correlation analysis
- k-means clustering
- hierarchical clustering
- dbscan
- silhouette score
- davies-bouldin index
- adjusted rand index (ari)
- monte carlo resampling
- cluster stability analysis

## customer variables

six quantitative variables are used in the clustering analysis:

- purchase frequency
- total purchase value
- total quantity purchased
- number of distinct products
- average purchase value
- average quantity per purchase

## repository files

- `PrimeiraParteTrabalhoFinal.py`: data preparation, exploratory analysis and initial clustering
- `SegundaParteTrabalhoFinal.py`: comparison of clustering methods and stability analysis
- `Online Retail.xlsx`: dataset used by the scripts
- `resultados_monte_carlo.csv`: exported monte carlo results
- `tabela_comparacao_metodos.csv`: comparison between clustering methods
- `relatorio.pdf`: final academic report
- `requirements.txt`: python dependencies

## how to run

keep all files in the same folder. the original python scripts were preserved without changes and therefore use relative paths based on this folder structure.

### 1. create a virtual environment

```bash
python -m venv .venv
```

### 2. activate the environment

windows powershell:

```bash
.venv\Scripts\Activate.ps1
```

windows command prompt:

```bash
.venv\Scripts\activate.bat
```

linux/mac:

```bash
source .venv/bin/activate
```

### 3. install the dependencies

```bash
pip install -r requirements.txt
```

### 4. run the first script

```bash
python PrimeiraParteTrabalhoFinal.py
```

this script generates `dados_etapa1.pkl`, which is required by the second part.

### 5. run the second script

```bash
python SegundaParteTrabalhoFinal.py
```

## notes

this repository contains an academic group project. the two original python scripts are intentionally kept unchanged from the submitted version.
