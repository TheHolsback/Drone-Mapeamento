# Definição do Filtro de Intensidade LIDAR

## Metodologia

Para a definição do filtro de erro ao obter uma medida, foi realizada uma **análise de bancada**. Foram escolhidos 4 ângulos de referência — 0°, 90°, 180° e 270° — com **1 minuto de amostragem** por ângulo.

### Valores de Referência Esperados

| Ângulo | Distância Esperada |
|-------:|-------------------:|
| 0°     | 1700 mm            |
| 90°    | 2000 mm            |
| 180°   | 1300 mm            |
| 270°   | 1000 mm            |

Ao final da coleta, foram obtidas **6584 aferições**.

---

## Análises Realizadas

Com os dados coletados, foram executadas as seguintes análises:

- Número de dados para cada ângulo
- Distribuição normal da distância aferida (por ângulo)
- Distribuição normal da intensidade (por ângulo)
- Correlação entre estar fora da curva da distância aferida e a intensidade

Para facilitar a análise dos resultados, foi desenvolvido o script [`lidar_relatorio.py`](lidar_relatorio.py).

---

## Resultados

Nos dois últimos gráficos gerados pelo script, observa-se uma tendência forte:

- **Gráfico da esquerda — Intensidade: Inliers vs Outliers:** os inliers se concentram predominantemente acima de 223 de intensidade, enquanto os outliers estão distribuídos por todas as faixas de intensidade — e quanto mais próximo de 0, maior a proporção de outliers.

- **Gráfico da direita — Curva de Retenção:** a curva é praticamente linear até a intensidade de **223**, após o qual há uma queda significativa na quantidade de dados retidos.

---

## Conclusão

Com base nessa análise, fica claro que a melhor escolha é **excluir da obtenção de dados os valores do LIDAR com intensidade menor que 223**.

> Este valor poderá ser revisado em análises posteriores, caso haja necessidade de ajuste.