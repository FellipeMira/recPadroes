# recPadroes

Este repositório contém scripts de workflow para reconhecimento de padrões.

A seleção de atributos agora utiliza o algoritmo Sequential Forward Floating Selection
(SFFS) implementado em `sffs.py`, dispensando o uso de RandomForest.

Além disso, o workflow aplica redução de dimensionalidade via PCA antes do
ajuste dos modelos.

O script inclui também uma implementação simplificada de um classificador do
tipo *RBF Network*, baseado em K-Means e regressão logística, que é otimizado
junto aos demais modelos.

O balanceamento das classes pode ser feito por `RandomUnderSampler` ou
`SMOTE`, ambos da biblioteca *imbalanced-learn*. Os valores-alvo (30.000,
38.694 e 3.452 amostras) são ajustados automaticamente para nunca exceder o
número de amostras disponíveis após a divisão do conjunto de treino no caso de
subamostragem.
Para definir qual deles será usado, altere a variável `SAMPLER_TYPE` em
`workflow3.py` para `"under"` ou `"smote"`.
