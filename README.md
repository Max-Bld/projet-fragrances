# Projet Fragrances

![Projet Fragrances image](https://i.ibb.co/db0K4Fy/dataset-cover.jpg "Projet Fragrances")

## Objectif

*L’objectif du projet Fragrances est de tenter d’identifier des classes de parfums à partir des ingrédients utilisés dans chacun d’eux à l'aide d'un algorithme K-Means non-supervisé.*

*Par exemple, les parfums qui partagent des notes de fruits seraient rassemblés dans la famille olfactive « fruitée », tandis que ceux où le musc est présent dans la famille « musquée ».*

## Ordre de consultation des fichiers

**projet_fragrances_dossier.pdf** décrit le déroulement complet du projet pour classifier ce jeu de données à l'aide de l'agorithme K-Means et des outils d'analyse textuelle de la librairie NLTK.

Les fichiers Python s'exécutent dans l'ordre suivant : **fragrances_1_eda.py**, **fragrances_2_pretraitement.py** et **fragrances_3_machinelearning.py**.

Le jeu de données de départ est **perfume_data_notes.csv**, et ceux d'arrivée qui contiennent les classes déterminées par l'algorithme K-Means sont **perfumes_3clusters.csv** et **perfumes_5clusters.csv**.

![Projet Fragrances Classification](https://i.ibb.co/d4TyLh5/image.png "Classification")
