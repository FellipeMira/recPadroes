# recPadroes

Este repositório contém scripts de workflow para reconhecimento de padrões.

A seleção de atributos agora utiliza o algoritmo Sequential Forward Floating Selection
(SFFS) implementado em `sffs.py`, dispensando o uso de RandomForest.

Além disso, o workflow aplica redução de dimensionalidade via PCA antes do
ajuste dos modelos.

O balanceamento das classes agora utiliza o `RandomUnderSampler` da
biblioteca *imbalanced-learn*, reduzindo a maioria para 30.000 instâncias
e mantendo 38.694 e 3.452 amostras nas demais classes.
