#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:30:49 2023

@author: maxbld
"""

import pandas as pd
from numpy import nan
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('df_norm.csv', index_col=0)

with open('unique_notes.txt', 'r') as unique_notes:
    lines = unique_notes.readlines()[0]

notes_set = eval(lines)
        
#%% Normalization

from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder().set_output(transform='pandas')
df_encoded = encoder.fit_transform(df).fillna(value=0)
df_norm = preprocessing.normalize(df_encoded)

#%% Fitting

kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')

kmeans.fit(df_norm)

kmeans.labels_

#%% Scoring

silhouette_score(df_norm, kmeans.labels_, metric='euclidean')

#%% Choosing best number of clusters

K = range(2, 8)
fits = []
score = []


for k in K:
    # train the model for current value of k on training data
    model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(df_norm)
    
    # append the model to fits
    fits.append(model)
    
    # Append the silhouette score to scores
    score.append(silhouette_score(df_norm, model.labels_, metric='euclidean'))

sns.lineplot(x = K, y = score)

#%%

sns.heatmap(data = df_encoded, cmap='viridis')


#%% Bag of words

from sklearn.feature_extraction.text import CountVectorizer

recette_df = pd.read_csv('df_recette.csv', index_col=0)

recette = recette_df['recette']


vectorizer = CountVectorizer()

vector = vectorizer.fit_transform(recette).todense()

voc = vectorizer.vocabulary_
print(voc)
len(voc)
print(vector)


#%%

from sklearn.metrics.pairwise import euclidean_distances


from sklearn.cluster import KMeans

vector_array = np.array(vector)

# euclidean_distances(vector_array)


kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
kmeans.fit(vector_array)

#%% BOW - choosing the best number of clusters

K = range(2, 8)
fits = []
score = []


for k in K:
    # train the model for current value of k on training data
    model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(vector_array)
    
    # append the model to fits
    fits.append(model)
    
    # Append the silhouette score to scores
    score.append(silhouette_score(vector_array, model.labels_, metric='euclidean'))

plt.figure(figsize=(10,6))
plt.title('SSE en fonction du nombre de clusters')
sns.lineplot(x = K, y = score)
plt.xlabel('Nombre de clusters')
plt.ylabel('SSE')
plt.grid()

#%% Visualisation

vector_df = pd.DataFrame(vector_array)
labels = pd.DataFrame({"label":kmeans.labels_})

df_labeled = pd.concat([df_encoded, labels], axis=1)
# df_labeled.rename(columns={0 : 'label'}, inplace=True)
# df_labeled['id'] = df_labeled.index
df_labeled = df_labeled.sort_values('label')
df_labeled = df_labeled.set_index('label')

plt.figure(figsize=(10, 10))
sns.heatmap(data = df_labeled, cmap = 'mako', cbar=False)
plt.title('K-Means Clustering')
plt.xlabel('Note')
plt.xticks(rotation = 45)
plt.ylabel('Class (cluster)')

df_category = pd.concat([recette, labels], axis=1)

#%% EXPORT
df_de_base = pd.read_csv('perfume_data_notes.csv')
df_and_label = pd.concat([df_de_base, labels], axis=1).to_csv('perfumes_3clusters.csv')
#%%

import nltk
from nltk.corpus import PlaintextCorpusReader

corpus_root = "/media/maxbld/2fba7426-4b05-4739-8f01-e170fb42b37b/home/maxbld/_data_max/pro/fragrances"
files = ".*\.txt"

corpus0 = PlaintextCorpusReader(corpus_root, files)
corpus  = nltk.Text(corpus0.words())

corpus.similar('leather')
corpus.common_contexts(['leather', 'musk'])
