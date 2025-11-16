# Projeto Individual (PI2): Clustering Não Supervisionado

## 🎯 Objetivo do Projeto

Este projeto tem como objetivo aplicar e comparar duas técnicas de **Aprendizado de Máquina Não Supervisionado** (Clustering) para resolver um problema de **Segmentação de Clientes Fictícios**. O foco é demonstrar o processo completo, desde a preparação dos dados (ETL) até a análise e interpretação dos resultados.

## 🛠️ Algoritmos Utilizados

Foram escolhidos dois algoritmos de clustering com metodologias distintas para a análise:

1.  **K-Means:** Algoritmo baseado em centroides, ideal para identificar grupos esféricos e bem separados.
2.  **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Algoritmo baseado em densidade, capaz de identificar clusters de formas arbitrárias e, crucialmente, detectar pontos de ruído (outliers).

## 📂 Estrutura do Repositório

| Arquivo | Descrição |
| :--- | :--- |
| `clustering_project.py` | Código-fonte principal contendo o ETL, a implementação dos modelos e a geração de gráficos. |
| `analise_e_resultados.md` | Documento detalhado com a descrição do problema, análise dos resultados e interpretação dos clusters. |
| `kmeans_elbow_method.png` | Gráfico do Método do Cotovelo para determinação do K ideal. |
| `kmeans_clusters.png` | Visualização dos clusters formados pelo K-Means. |
| `dbscan_clusters.png` | Visualização dos clusters formados pelo DBSCAN (incluindo ruído). |
| `README.md` | Este arquivo de documentação. |

## 🚀 Como Executar o Projeto

Siga os passos abaixo para configurar o ambiente e executar o script Python.

### 1. Requisitos

Certifique-se de ter o **Python 3.x** instalado em seu sistema.

### 2. Configuração do Ambiente Virtual (venv)

É altamente recomendável utilizar um ambiente virtual para isolar as dependências do projeto.

```bash
# 1. Criar o ambiente virtual
python3 -m venv venv

# 2. Ativar o ambiente virtual
# No Linux/macOS:
source venv/bin/activate

# No Windows (Command Prompt):
# venv\Scripts\activate
```

### 3. Instalação das Dependências

Com o ambiente virtual ativado, instale as bibliotecas necessárias:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 4. Execução do Script

Execute o arquivo principal. Ele irá gerar os dados fictícios, treinar os modelos, imprimir a análise no console e salvar os gráficos e o arquivo de análise (`analise_e_resultados.md`) no diretório raiz.

```bash
python clustering_project.py
```

## 📊 Resultados e Análise

O script gera automaticamente os seguintes artefatos visuais e de documentação:

1.  **`kmeans_elbow_method.png`**: Demonstra a heurística utilizada para escolher o número de clusters (K=4).
2.  **`kmeans_clusters.png`**: Mostra a segmentação clara dos 4 grupos de clientes.
3.  **`dbscan_clusters.png`**: Ilustra a capacidade do DBSCAN de encontrar clusters baseados em densidade e identificar outliers (pontos de ruído).

Para a interpretação detalhada de cada cluster e a comparação entre os modelos, consulte o arquivo **`analise_e_resultados.md`**.
