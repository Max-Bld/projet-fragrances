#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:37:46 2023

@author: maxbld
"""


import pandas as pd
import re


df = pd.read_csv('/home/maxbld/_data_max/pro/fragrance/perfume_data_notes.csv')

#%% 2. Pré-traitement Machine Learning
#%%% a. Suppression des colonnes inutiles

notes_df = df.drop(['brand', 'title', 'gender'], axis=1)

#%%% b. Standardisation des données

    # Remplacement des Na par un espace

notes_df = notes_df.fillna(value=" ")


    # Mettre en minuscule

notes_df = notes_df.apply(lambda x : x.str.lower())


    # Supprimer les caractères spéciaux
    
for to_be_replaced in ["(", ")", "[", "]"]:
    notes_df = notes_df.apply(lambda x : x.str.replace(to_be_replaced, ""))


    # Supprimer les espaces

pattern = "\s"                
regexp = re.compile(pattern)

for n in notes_df.columns:
    notes_df[n] = notes_df[n].apply(lambda x : re.sub(pattern, "", x))


    # Supprimer les caractères numériques

pattern = "\d"                      
regexp = re.compile(pattern)

for n in notes_df.columns:
    notes_df[n] = notes_df[n].apply(lambda x : re.sub(pattern, "", x))


#%%% Exports

    # Jeu de données standardisé

notes_df.to_csv('df_norm.csv')


    # Jeu de données contenant seulement la recette concaténée
    
        # Concaténation

notes_df['recette'] = notes_df['note1'] + " " + notes_df['note2'] + " " + notes_df['note3'] + " " + notes_df['note4'] + " " + notes_df['note5'] + " " + notes_df['note6'] + " " + notes_df['note7'] + " " + notes_df['note8'] + " " + notes_df['note9'] + " " + notes_df['note10'] + " " + notes_df['note11'] + " " + notes_df['note12'] + " " + notes_df['note13'] + " " + notes_df['note14'] + " " + notes_df['note15'] + " " + notes_df['note16'] + " " + notes_df['note17'] + " " + notes_df['note18'] + " " + notes_df['note19'] + " " + notes_df['note20']
recette_df = notes_df['recette']

        # Export en .csv
        
recette_df.to_csv('df_recette.csv')

#%%% Export en .txt pour traitement NLTK

    # Attention : cette ligne crée plus de 30 000 fichiers .txt

for n in range(len(recette_df)):
    f = open(f'/media/maxbld/2fba7426-4b05-4739-8f01-e170fb42b37b/home/maxbld/_data_max/pro/fragrances/{n}.txt', 'w')
    f.write(str(recette_df[n]))
    f.close()

