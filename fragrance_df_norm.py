#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:38:05 2023

@author: maxbld
"""

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('perfume_data_notes.csv')

with open('unique_notes.txt', 'r') as unique_notes:
    lines = unique_notes.readlines()[0]

notes_set = eval(lines)

notes_df = df.drop(['brand', 'title', 'gender'], axis=1)


notes_df = notes_df.fillna(value=" ")

notes_df = notes_df.apply(lambda x : x.str.lower())

for to_be_replaced in ["(", ")", "[", "]"]:
    notes_df = notes_df.apply(lambda x : x.str.replace(to_be_replaced, ""))
    
pattern = "\s"                  # delete spaces
regexp = re.compile(pattern)

for n in notes_df.columns:
    notes_df[n] = notes_df[n].apply(lambda x : re.sub(pattern, "", x))
    
pattern = "\d"                      # suppress digits
regexp = re.compile(pattern)

for n in notes_df.columns:
    notes_df[n] = notes_df[n].apply(lambda x : re.sub(pattern, "", x))

notes_df.to_csv('df_norm.csv')


#%% Count unique 

notes_list=[]
for n in notes_df:
    notes_list.append(notes_df[n].unique().tolist())

notes = []
for lst in notes_list:
        for n in lst:
            notes.append(n)
            
unique_notes = list(set(notes))
                   
print(f"{df.brand.nunique()} unique brands.")
print(f"{df.title.nunique()} unique perfumes.")
print(f"{len(unique_notes)} unique ingredients.")

#%%


notes_df['recette'] = notes_df['note1'] + " " + notes_df['note2'] + " " + notes_df['note3'] + " " + notes_df['note4'] + " " + notes_df['note5'] + " " + notes_df['note6'] + " " + notes_df['note7'] + " " + notes_df['note8'] + " " + notes_df['note9'] + " " + notes_df['note10'] + " " + notes_df['note11'] + " " + notes_df['note12'] + " " + notes_df['note13'] + " " + notes_df['note14'] + " " + notes_df['note15'] + " " + notes_df['note16'] + " " + notes_df['note17'] + " " + notes_df['note18'] + " " + notes_df['note19'] + " " + notes_df['note20']
recette_df = notes_df['recette']

for n in range(len(recette_df)):
    f = open(f'/media/maxbld/2fba7426-4b05-4739-8f01-e170fb42b37b/home/maxbld/_data_max/pro/fragrances/{n}.txt', 'w')
    f.write(str(recette_df[n]))
    f.close()

recette_df.to_csv('df_recette.csv')

#%% Multiple Files

from nltk.corpus import PlaintextCorpusReader
import nltk

corpus_root = "/media/maxbld/2fba7426-4b05-4739-8f01-e170fb42b37b/home/maxbld/_data_max/pro/fragrances"
files = ".*\.txt"

corpus0 = PlaintextCorpusReader(corpus_root, files)
corpus  = nltk.Text(corpus0.words())

corpus.similar('bulgarianrose')

#%% Single File

corpus_root = "/media/maxbld/2fba7426-4b05-4739-8f01-e170fb42b37b/home/maxbld/_data_max/pro/fragrances/recette"
wordlists = PlaintextCorpusReader(corpus_root, '.*')
text1 = wordlists.open('/media/maxbld/2fba7426-4b05-4739-8f01-e170fb42b37b/home/maxbld/_data_max/pro/fragrances/recette/df_recette.txt')