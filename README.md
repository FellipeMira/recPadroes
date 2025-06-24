# recPadroes

Este repositório contém scripts de workflow para reconhecimento de padrões.

A seleção de atributos agora utiliza o algoritmo Sequential Forward Floating Selection
(SFFS) implementado em `sffs.py`, dispensando o uso de RandomForest.

Além disso, o workflow aplica redução de dimensionalidade via PCA antes do
ajuste dos modelos.

O script inclui também uma implementação simplificada de um classificador do
tipo *RBF Network*, baseado em K-Means e solução analítica via pseudoinversa, que é otimizado
junto aos demais modelos.

O balanceamento das classes pode ser feito por `RandomUnderSampler` ou
`SMOTE`, ambos da biblioteca *imbalanced-learn*. Os valores-alvo (33.993
ou 3.452 amostras, dependendo da região) são ajustados automaticamente para
nunca exceder o número de amostras disponíveis após a divisão do conjunto de
treino no caso de subamostragem.

Para definir qual deles será usado, altere a variável `SAMPLER_TYPE` em
`workflow3.py` para `"under"` ou `"smote"`.

O dataset utilizado possui rótulos `-1`, `0` e `1`. Agora os registros com rótulo
`0` são descartados e o mapeamento passa a ser `{-1: 1, 1: 0}`.

Dois scripts foram adicionados:

* `train_models.py` – treina os modelos, salva cada um em `model_SFFS_*` e
  `model_PCA_*` e grava as métricas de validação.
* `predict_models.py` – carrega os modelos treinados, aplica as mesmas
  transformações e gera um CSV com as previsões.

Para acelerar os modelos do scikit-learn, o script utiliza a biblioteca
`scikit-learn-intelex` (importada via `sklearnex`). Caso ela não esteja
instalada, execute `pip install scikit-learn-intelex` antes de rodar o
workflow.
