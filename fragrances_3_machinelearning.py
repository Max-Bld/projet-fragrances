#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:48:57 2023

@author: maxbld
"""

import time

import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import nltk
from nltk.corpus import PlaintextCorpusReader


df_norm = pd.read_csv('/home/maxbld/_data_max/pro/fragrance/df_norm.csv', index_col=0)
df_recette = pd.read_csv('/home/maxbld/_data_max/pro/fragrance/df_recette.csv', index_col=0)

#%% 3. Machine Learning : classification non-supervisée K-means
#%%% a. Choix : bag-of-words
#%%% b. Encodage des données

vectorizer = CountVectorizer()

vector = vectorizer.fit_transform(df_recette['recette']).todense()

vocabulaire = vectorizer.vocabulary_

print(f"Vocabulaire de {len(vocabulaire)} mots uniques associés à une valeur unique :")
time.sleep(1)
print(vocabulaire)


time.sleep(1)
print("Vecteur bag-of-words de la forme :")
time.sleep(1)
print(vector)

#%%% c. Classification K-means

    # Entrainement

vector_array = np.array(vector)

kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
kmeans.fit(vector_array)


    # Visualisation
    
        # Ordinal Encoding pour la visualisation
        
encoder = OrdinalEncoder().set_output(transform='pandas')
df_encoded = encoder.fit_transform(df_norm).fillna(value=0)


        # Concaténation des DF et Series désirés
        
vector_df = pd.DataFrame(vector_array)
labels = pd.DataFrame({"label":kmeans.labels_})
df_labeled = pd.concat([df_encoded, labels], axis=1)


        # Transformation du DF
        
df_labeled = df_labeled.sort_values('label')
df_labeled = df_labeled.set_index('label')


        # Heatmap
        
plt.figure(figsize=(10, 10))
sns.heatmap(data = df_labeled, cmap = 'mako', cbar=False)
plt.title('K-Means Clustering (3 clusters)')
plt.xlabel('Note')
plt.xticks(rotation = 45)
plt.ylabel('Class (cluster)')

        # Export
        
df_de_base = pd.read_csv('/home/maxbld/_data_max/pro/fragrance/perfume_data_notes.csv')
df_and_label = pd.concat([df_de_base, labels], axis=1).to_csv('perfumes_3clusters.csv')

#%%% d. Analyse textuelle avec NLTK

    # Chemin des textes exportés depuis fragrances_pretraitement.py

corpus_root = "/media/maxbld/2fba7426-4b05-4739-8f01-e170fb42b37b/home/maxbld/_data_max/pro/fragrances"
files = ".*\.txt"

corpus0 = PlaintextCorpusReader(corpus_root, files)
corpus  = nltk.Text(corpus0.words())

corpus.similar('leather')
corpus.similar('musk')

#%%% f. Optimisation des hyper-paramètres


    # Test du comportement de l'algorithme selon un intervalle K de clusters différents
    # (Peut durer plusieurs minutes)

K = range(2, 8)
fits = []
score = []


for k in K:
        # Entrainer le modèle à la valeur k sur le jeu de données
    model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(vector_array)
    
        # Ajouter le modèle à fits
    fits.append(model)
    
        # Ajouter le silhouette score à score
    score.append(silhouette_score(vector_array, model.labels_, metric='euclidean'))


    # Tracer le silhouette score en fonction du nombre de cluster
    # (Méthode Elbow)

plt.figure(figsize=(10,6))
plt.title('SSE en fonction du nombre de clusters')
sns.lineplot(x = K, y = score)
plt.xlabel('Nombre de clusters')
plt.ylabel('SSE')
plt.grid()


#%% K-means à 5 clusters

kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto')
kmeans.fit(vector_array)


    # Visualisation

vector_df = pd.DataFrame(vector_array)
labels = pd.DataFrame({"label":kmeans.labels_})
df_labeled = pd.concat([df_encoded, labels], axis=1)
        
df_labeled = df_labeled.sort_values('label')
df_labeled = df_labeled.set_index('label')
        
plt.figure(figsize=(10, 10))
sns.heatmap(data = df_labeled, cmap = 'mako', cbar=False)
plt.title('K-Means Clustering (5 clusters)')
plt.xlabel('Note')
plt.xticks(rotation = 45)
plt.ylabel('Class (cluster)')


    # Export des résultats

df_de_base = pd.read_csv('/home/maxbld/_data_max/pro/fragranceperfume_data_notes.csv')
df_and_label = pd.concat([df_de_base, labels], axis=1).to_csv('perfumes_5clusters.csv')