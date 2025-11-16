
## Projeto Individual (PI2): Clustering Não Supervisionado

**Problema:** Segmentação de Clientes Fictícios.

O objetivo deste projeto é aplicar técnicas de aprendizado de máquina não supervisionado para identificar grupos (clusters) de clientes com base em seus hábitos de compra e renda. A segmentação de clientes é crucial para estratégias de marketing personalizadas, permitindo que a empresa direcione ofertas específicas para cada grupo, otimizando o retorno sobre o investimento (ROI).

**Algoritmos Escolhidos:**
1. **K-Means:** Um algoritmo de clustering baseado em centroides, eficiente e amplamente utilizado.
2. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Um algoritmo baseado em densidade, capaz de encontrar clusters de formas arbitrárias e identificar ruído (outliers).


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
