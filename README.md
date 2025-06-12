# recPadroes

Este repositório contém scripts de workflow para reconhecimento de padrões.

A seleção de atributos agora utiliza o algoritmo Sequential Forward Floating Selection
(SFFS) implementado em `sffs.py`, dispensando o uso de RandomForest.

Além disso, o workflow aplica redução de dimensionalidade via PCA antes do
ajuste dos modelos.
