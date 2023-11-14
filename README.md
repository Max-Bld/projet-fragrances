# Projet Fragrances
ML: Non-supervised K-Means classification of perfumes

![Projet Fragrances image](https://i.ibb.co/db0K4Fy/dataset-cover.jpg "Projet Fragrances")

## Objectif

*L’objectif du projet Fragrances est de tenter d’identifier des classes de parfums à partir des ingrédients utilisés dans chacun d’eux.*

*Par exemple, les parfums qui partagent des notes de fruits seraient rassemblés dans la famille olfactive « fruitée », tandis que ceux où le musc est présent dans la famille « musquée ».*

## Ordre de consultation des fichiers

Le fichier à consulter en premier est **projet_fragrances_dossier.pdf** car il décrit le déroulement complet du projet pour classifier ce jeu de données à l'aide de l'agorithme K-Means et des outils d'analyse textuelle de la librairie NLTK.

Le jeu de données de départ est **perfume_data_notes.csv**, et ceux d'arrivée qui contiennent les classes déterminées par l'algorithme K-Means sont **perfumes_3clusters.csv** et **perfumes_5clusters.csv**.

Les fichiers Python s'exécutent dans l'ordre suivant : **fragrance.py**, **fragrance_df_norm.py** et **fragrance_numerisation.py**.

Une mise à jour de ce repository viendra prochainement pour rendre plus clairs les fichiers Python.
