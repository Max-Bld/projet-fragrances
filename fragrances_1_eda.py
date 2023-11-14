#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:53:50 2023

@author: maxbld
"""
import time

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt


df = pd.read_csv('/home/maxbld/_data_max/pro/fragrance/perfume_data_notes.csv')

#%% 1. Analyse exploratoire des données
#%%% a. Caractéristiques générales

    # Informations générales

print("Dimensions du jeu de données :", df.shape)
time.sleep(1)
print("\nColonnes du jeu de données :")
for n in df.columns:
    print(n)
    time.sleep(0.1)
time.sleep(1)

    # Valeurs uniques de chaque colonne

all_notes = []
for col in df.columns[2:22]:
    for note in df[col] : 
        all_notes.append(note)
unique_notes = set(all_notes)

print(f"\n{df.brand.nunique()} unique brands.")
time.sleep(0.75)
print(f"{df.title.nunique()} unique perfumes.")
time.sleep(0.75)
print(f"{len(unique_notes)} unique notes (ingredients).")
time.sleep(0.75)
print(f"{df.gender.nunique()} unique genders.")


    # Tableau récapitulatif

print("\nInformations des colonnes du jeu de données :")
print(df.describe())
time.sleep(1)

#%%% b. Distribution des valeurs manquantes

    # Matrice valeurs manquantes
    
plt.figure(figsize=(8,8))
sns.heatmap(df.isnull(), cbar=False)
plt.xlabel('Colonnes')
plt.xticks(rotation=45)
plt.ylabel('Index')
plt.title('MATRICE DES VALEURS MANQUANTES \n (représentées en blanc)')

#%%% c. Distribution détaillée des colonnes

    # Colonnes note1 à note20
    
notes_df = df.drop(['brand', 'title', 'gender'], axis=1)

notes_df = (notes_df.isna() == False).sum()

plt.figure(figsize = (9,6))
plt.bar(
        notes_df.index, 
        height = notes_df,
        color = sns.color_palette('mako_r', len(notes_df.index))
        )
plt.title('Distribution des notes (ingrédients)')
plt.ylabel("Count")
plt.xticks(rotation = 45)

    # Colonne gender
    
df_gb = df[['gender', 'brand']].groupby(['gender']).count()
df_gb.loc['nan'] = len(df) - df_gb.sum()

plt.figure(figsize = (8,5))
plt.bar(
        x = df_gb.index, 
        height = df_gb['brand'],
        color = sns.color_palette('mako_r', 3)
        )
plt.title("Distribution de la colonne 'gender'")
plt.xlabel("Catégories de gender")
plt.ylabel("Count")
