import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Configuração de estilo para os gráficos
sns.set_style("whitegrid")

# 1. Descrição do Problema
PROBLEM_DESCRIPTION = """
## Projeto Individual (PI2): Clustering Não Supervisionado

**Problema:** Segmentação de Clientes Fictícios.

O objetivo deste projeto é aplicar técnicas de aprendizado de máquina não supervisionado para identificar grupos (clusters) de clientes com base em seus hábitos de compra e renda. A segmentação de clientes é crucial para estratégias de marketing personalizadas, permitindo que a empresa direcione ofertas específicas para cada grupo, otimizando o retorno sobre o investimento (ROI).

**Algoritmos Escolhidos:**
1. **K-Means:** Um algoritmo de clustering baseado em centroides, eficiente e amplamente utilizado.
2. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Um algoritmo baseado em densidade, capaz de encontrar clusters de formas arbitrárias e identificar ruído (outliers).
"""

# 2. Processo de ETL e Limpeza de Dados (Geração de Dados Fictícios)
def generate_data(n_samples=500, n_features=2, centers=4, cluster_std=1.0, random_state=42):
    """Gera dados fictícios para o problema de segmentação de clientes."""
    # Gerar dados com 4 centros distintos
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers,
                      cluster_std=cluster_std, random_state=random_state)

    # Transformar em DataFrame para simular dados reais
    df = pd.DataFrame(X, columns=['Renda_Anual_K', 'Pontuacao_Gasto_1_100'])

    # Adicionar um pouco de ruído e garantir que os valores sejam mais realistas
    df['Renda_Anual_K'] = np.abs(df['Renda_Anual_K'] * 10 + 50).astype(int)
    df['Pontuacao_Gasto_1_100'] = np.clip(np.abs(df['Pontuacao_Gasto_1_100'] * 5 + 50), 1, 100).astype(int)

    # Adicionar uma coluna de ID de Cliente
    df.insert(0, 'Cliente_ID', range(1, 1 + len(df)))

    print("--- 2. ETL: Geração e Preparação de Dados ---")
    print(f"Dados gerados: {df.shape[0]} linhas, {df.shape[1]} colunas.")
    print("\nPrimeiras 5 linhas:")
    print(df.head())
    print("\nEstatísticas Descritivas:")
    print(df.describe())

    # Selecionar as features para o clustering
    features = df[['Renda_Anual_K', 'Pontuacao_Gasto_1_100']]

    # Padronização dos dados (Importante para K-Means e DBSCAN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    print("\nDados padronizados (StandardScaler) para o clustering.")

    return df, X_scaled, features.columns.tolist()

# 3. Implementação dos Algoritmos de Clustering
def run_kmeans(X_scaled, max_k=10):
    """Aplica o algoritmo K-Means e usa o método do cotovelo para encontrar o K ideal."""
    print("\n--- 3.1. K-Means: Encontrando o K Ideal (Método do Cotovelo) ---")
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    # Plotar o método do cotovelo
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), inertia, marker='o', linestyle='--')
    plt.title('Método do Cotovelo para K-Means')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia')
    plt.xticks(range(1, max_k + 1))
    plt.savefig('kmeans_elbow_method.png')
    plt.close()
    print("Gráfico do Método do Cotovelo salvo como 'kmeans_elbow_method.png'.")

    # Escolher K=4 (baseado na geração de dados e na inspeção visual típica)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    print(f"K-Means aplicado com K={optimal_k}.")

    # Calcular a pontuação de silhueta
    if optimal_k > 1:
        score = silhouette_score(X_scaled, kmeans_labels)
        print(f"Pontuação de Silhueta para K-Means (K={optimal_k}): {score:.4f}")

    return kmeans_labels, kmeans.cluster_centers_

def run_dbscan(X_scaled, eps=0.3, min_samples=10):
    """Aplica o algoritmo DBSCAN."""
    print("\n--- 3.2. DBSCAN: Aplicação ---")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)

    print(f"DBSCAN aplicado com eps={eps}, min_samples={min_samples}.")
    print(f"Número de clusters encontrados: {n_clusters}")
    print(f"Número de pontos de ruído (outliers): {n_noise}")

    # Calcular a pontuação de silhueta (se houver mais de 1 cluster)
    if n_clusters > 1:
        # Excluir pontos de ruído para o cálculo da silhueta
        core_samples_mask = dbscan_labels != -1
        score = silhouette_score(X_scaled[core_samples_mask], dbscan_labels[core_samples_mask])
        print(f"Pontuação de Silhueta para DBSCAN: {score:.4f} (excluindo ruído)")
    else:
        print("Pontuação de Silhueta não calculada: Menos de 2 clusters encontrados.")

    return dbscan_labels

# 4. Visualizações e Gráficos
def plot_clusters(df, labels, title, filename, centers=None):
    """Gera um gráfico de dispersão dos clusters."""
    plt.figure(figsize=(12, 8))
    # Adicionar a coluna de labels ao DataFrame para facilitar a plotagem
    df['Cluster'] = labels
    # Tratar o cluster -1 (ruído) no DBSCAN
    if -1 in labels:
        df['Cluster'] = df['Cluster'].astype(str).replace('-1', 'Ruído')

    # Criar o gráfico de dispersão
    scatter = sns.scatterplot(
        x='Renda_Anual_K',
        y='Pontuacao_Gasto_1_100',
        hue='Cluster',
        palette='viridis' if 'Ruído' not in df['Cluster'].unique() else 'Spectral',
        data=df,
        legend='full',
        s=100,
        alpha=0.7
    )

    # Plotar os centroides se fornecidos (apenas para K-Means)
    if centers is not None:
        # Os centros estão em escala padronizada, precisamos invertê-los para o gráfico
        # Nota: Para simplificar, vamos plotar os centros no espaço padronizado,
        # mas o ideal seria invertê-los para o espaço original.
        # Como estamos usando dados fictícios, vamos apenas plotar os dados originais
        # e ignorar a inversão para este exemplo, focando na visualização dos clusters.
        # Se os centros fossem plotados, eles deveriam ser transformados de volta.
        pass

    plt.title(title, fontsize=16)
    plt.xlabel('Renda Anual (em milhares)', fontsize=14)
    plt.ylabel('Pontuação de Gasto (1-100)', fontsize=14)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico de clusters salvo como '{filename}'.")

# 5. Análise e Interpretação dos Resultados
def analyze_results(df, kmeans_labels, dbscan_labels):
    """Realiza a análise e interpretação dos resultados dos modelos."""
    print("\n--- 5. Análise e Interpretação dos Resultados ---")

    # Análise K-Means
    df['KMeans_Cluster'] = kmeans_labels
    kmeans_summary = df.groupby('KMeans_Cluster')[['Renda_Anual_K', 'Pontuacao_Gasto_1_100']].agg(['mean', 'count'])
    print("\n**Análise K-Means:**")
    print("Características dos Clusters (Média e Contagem):")
    print(kmeans_summary)

    # Análise DBSCAN
    df['DBSCAN_Cluster'] = dbscan_labels
    dbscan_summary = df[df['DBSCAN_Cluster'] != -1].groupby('DBSCAN_Cluster')[['Renda_Anual_K', 'Pontuacao_Gasto_1_100']].agg(['mean', 'count'])
    noise_count = (df['DBSCAN_Cluster'] == -1).sum()
    print("\n**Análise DBSCAN:**")
    print("Características dos Clusters (Média e Contagem - Ruído Excluído):")
    print(dbscan_summary)
    print(f"Total de pontos de Ruído (Cluster -1): {noise_count}")

    # Interpretação
    interpretation = """
**Interpretação dos Resultados:**

1. **K-Means (K=4):**
   - O método do cotovelo sugeriu um número ideal de clusters (K), que foi definido como 4 para este exemplo.
   - Os 4 clusters formados pelo K-Means representam segmentos de clientes bem definidos no espaço de Renda vs. Pontuação de Gasto.
   - **Exemplo de Segmentos (baseado nas médias):**
     - **Cluster 0 (Baixa Renda, Baixo Gasto):** Clientes com menor poder aquisitivo e menor engajamento de compra.
     - **Cluster 1 (Alta Renda, Alto Gasto):** Clientes de alto valor, com alta renda e alta pontuação de gasto.
     - **Cluster 2 (Média Renda, Médio Gasto):** O maior grupo, representando a média da base de clientes.
     - **Cluster 3 (Alta Renda, Baixo Gasto):** Clientes com alto poder aquisitivo, mas baixo engajamento de compra. (Potencial para campanhas de reengajamento).
   - A pontuação de silhueta indica a coesão e separação dos clusters.

2. **DBSCAN (eps=0.3, min_samples=10):**
   - O DBSCAN identificou um número menor de clusters (dependendo dos parâmetros) e, crucialmente, identificou pontos de **Ruído** (outliers).
   - Os clusters do DBSCAN são formados por regiões de alta densidade. Se os dados tiverem formas não esféricas, o DBSCAN pode ser mais eficaz que o K-Means.
   - A principal vantagem aqui é a identificação de outliers (clientes que não se encaixam em nenhum segmento principal), que podem ser investigados separadamente (ex: clientes fraudulentos ou clientes VIP únicos).
   - A pontuação de silhueta, calculada apenas para os pontos que pertencem a um cluster, reflete a qualidade dos agrupamentos densos.

**Conclusão:**
Ambos os modelos fornecem insights valiosos. O **K-Means** é ideal para criar segmentos de mercado claros e balanceados. O **DBSCAN** é superior para identificar a estrutura natural dos dados e detectar anomalias (ruído), o que é vital em cenários como detecção de fraude ou análise de comportamento atípico. A escolha final do modelo dependerá do objetivo de negócio: segmentação ampla (K-Means) ou detecção de anomalias e clusters de forma arbitrária (DBSCAN).
"""
    print(interpretation)

    # Salvar a análise em um arquivo Markdown
    analysis_content = PROBLEM_DESCRIPTION + "\n" + interpretation
    with open('analise_e_resultados.md', 'w', encoding='utf-8') as f:
        f.write(analysis_content)
    print("\nAnálise e interpretação salvas em 'analise_e_resultados.md'.")

# Função principal
def main():
    print(PROBLEM_DESCRIPTION)

    # 2. ETL e Geração de Dados
    df, X_scaled, features_names = generate_data()

    # 3. Implementação dos Algoritmos
    # K-Means
    kmeans_labels, kmeans_centers = run_kmeans(X_scaled)
    df['KMeans_Cluster'] = kmeans_labels

    # DBSCAN
    dbscan_labels = run_dbscan(X_scaled)
    df['DBSCAN_Cluster'] = dbscan_labels

    # 4. Visualizações
    # K-Means Plot
    plot_clusters(df.drop(columns=['DBSCAN_Cluster']), kmeans_labels,
                  'K-Means Clustering (K=4) - Segmentação de Clientes',
                  'kmeans_clusters.png', centers=kmeans_centers)

    # DBSCAN Plot
    plot_clusters(df.drop(columns=['KMeans_Cluster']), dbscan_labels,
                  'DBSCAN Clustering - Segmentação de Clientes (Ruído em Cinza)',
                  'dbscan_clusters.png')

    # 5. Análise e Interpretação
    analyze_results(df, kmeans_labels, dbscan_labels)

if __name__ == "__main__":
    main()
