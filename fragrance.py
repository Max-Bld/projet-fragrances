#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 17:34:49 2023

@author: maxbld
"""

#%% EDA

import pandas as pd
import seaborn as sns
import numpy as np
# import _data_max.pro.fragrance.personalfunctions as pf
import re

df = pd.read_csv('perfume_data_notes.csv')
df2 = pd.read_csv('perfume_data_links.csv')
# summary = pd.DataFrame(index = ["total", "unique_brands", "unique_perfumes", "unique_ingredients"])
notes_df = df.drop(['brand', 'title', 'gender'], axis=1)

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

#%% Cleaning

unique_notes = list(set(notes))
unique_notes_new = []


for note in unique_notes:
    if type(note) == float:
        unique_notes.remove(note)        
        
    else:
        if note[-1] == " ":
            note = note[:-1]
        
        for to_be_replaced in ["(", ")", "[", "]"]:
            note = note.replace(to_be_replaced, "")
            
        
        
        note = note.lower()                 # lowercase
        
        pattern = "\s{1,}"                  # replace multiple space by only one space
        regexp = re.compile(pattern)
        
        note = re.sub(pattern, "_", note)
        
        pattern = "\d"                      # suppress digits
        regexp = re.compile(pattern)
        
        note = re.sub(pattern, "", note)
        
        unique_notes_new.append(note)
     
unique_notes_new = set(unique_notes_new)

with open("unique_notes.txt", "w") as output:
    output.write(str(unique_notes_new))

print(f"{len(unique_notes_new)} ingrédients standardisés.")

#%% Clustering words (SEMANTIC APPROACH)

"""
Semantic approach is not conclusive and is abandoned to a whole-word
approach.
"""

split = []
unique_notes_split=[]
unique_notes_element_count_list = []

for word in (set(unique_notes_new)):    # split based on space character
    split.append(word.split(" "))
    for element in split[-1]:
        unique_notes_split.append(element)
        
print(f"{len(unique_notes_split)} mots (non-uniques) d'ingrédients.")
print(f"{len(set(unique_notes_split))} mots (uniques) d'ingrédients.")

for word in set(unique_notes_split):
    unique_notes_element_count_list.append((word, unique_notes_split.count(word)))

df_words = pd.DataFrame(unique_notes_element_count_list, columns = ["word", "number"])

df_words_gb = df_words.groupby('word')['number'].sum().sort_values(ascending=False)
len(df_words_gb)
df_words_gb.sum()

sns.barplot(x=df_words_gb.index[:10], y=df_words_gb[:10], errorbar=None, )

df_words_gb[:10].sum()

    #%%
for element in unique_notes_split:      # count occurences of each splitted word in the list and couple these in a tuple
    unique_notes_element_count_list.append((element, unique_notes_split.count(element)))

elements_df = pd.DataFrame(unique_notes_element_count_list, columns=['name', 'count'])  # df enables the useful groupby function

element_count = elements_df.groupby('name')['count'].sum().sort_values(ascending=False)
element_count

gb_elements = elements_df.groupby('name')['count'].sum()

gb_elements.sum()

total = 0
for n in unique_notes_element_count_list:
    total = total + n[1]
    
elements_df['count'].sum()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù


#%%% df notes exploration
import matplotlib.pyplot as plt

plt.figure(figsize=(7,7))
sns.heatmap(df.isnull(), cbar=False)
plt.xlabel('Colonnes')
plt.ylabel('Index')
plt.title('MATRICE DES VALEURS MANQUANTES \n (représentées en blanc)')

# desc_df = pf.describe_fully(df)

# print(desc_df)

df_selected['note1'].loc[0]

unique_brands_df= desc_df['describe'].loc['unique'][0] # 3123
unique_perfumes_df=desc_df['describe'].loc['unique'][1] # 31077
df_selected = df.drop(['brand', 'title', 'gender'], axis = 1) # only notes cols
dic_unique_df1=pf.aggregate_unique(df_selected)
unique_ingredients_df=len(dic_unique_df1) # 1416

summary['df1']= [len(df), unique_brands_df, unique_perfumes_df, unique_ingredients_df]
#%%

notes_df = df.drop(['brand', 'title', 'gender'], axis=1)

notes_df = (notes_df.isna() == False).sum()

plt.figure(figsize = (9,6))
plt.bar(notes_df.index, height = notes_df, color = sns.color_palette('mako_r', len(notes_df.index)))
plt.title('Distribution des notes (ingrédients)')
plt.ylabel("Count")
plt.xticks(rotation = 45)

#%% Distrib gender

df_gb = df[['gender', 'brand']].groupby(['gender']).count()
df_gb.loc['nan'] = len(df) - df_gb.sum()

df_gb.loc['man'] / len(df) * 100
df_gb.loc['unisex'] / len(df) * 100
df_gb.loc['women'] / len(df) * 100
df_gb.loc['nan'] / len(df) * 100

plt.figure(figsize = (8,5))
plt.title("Distribution de la colonne 'gender'")
plt.xlabel("Catégories de gender")
plt.ylabel("Count")
plt.bar(x = df_gb.index, 
        height = df_gb['brand'],
        color = sns.color_palette('mako_r', 3)
        )
#%%

for rect1 in p1:
    height = int(rect1.get_height())/total * 100
    plt.annotate( f"{height:.2f}%".format(height),(rect1.get_x() + rect1.get_width()/2, height+.05),ha="center",va="bottom",fontsize=15)


#%%% df_links exploration

desc_df2 = pf.describe_fully(df2)
print(desc_df2)

unique_brands_df2=desc_df2['describe'].loc['unique'][0]
unique_perfumes_df2=desc_df2['describe'].loc['unique'][3]
unique_ingredients_df2=desc_df2['describe'].loc['unique'][-1]

summary['df2']=[len(df2), unique_brands_df2, unique_perfumes_df2, unique_ingredients_df2]

#%%

pf.display_unique(desc_df2['describe'])   


anka=df2[df2['Perfume URL']=='https://www.parfumo.com/Perfumes/anka-kus-parfum/the-long-red-cloud'] # see the perfume with biggest number of notes

print(anka) # 70 notes

#%%% df notes transformation

for x in df.columns:
    df[x]=df[x].str.lower()

#%%  
dic_df = {}

for x in df.columns[2:22]:  #put all notes in one col
    dic_df[x]= df[['brand', 'title', 'gender', x]]
    dic_df[x].rename(columns={dic_df[x].columns[-1]:'note'}, inplace=True)

dic_df
    
    
#%%
df = df.drop('all_notes', axis=1)
df.where(df.isnull())
#%%% df links transformation
for x in df2.columns:
    df2[x]=df2[x].str.lower()
    
df2_selected = df2.Note
dic_unique_df2 = pf.aggregate_unique(df2_selected)


df2_gb=df2[['Perfume URL', 'Note']].groupby('Perfume URL').count()

df2['index_col']=df2.index
df2_gb = df2.groupby('Perfume Name').count()

df2_new = pd.DataFrame()

for w in range(len(df2)):
    for x in df2_gb['Brand'].index:
        for y in range(x):
            df2_new[f'note{y}'] = df2.Note.iloc[w+y]
            
            
#%%

df2_test = df2.iloc[:100]
df2_test
dic_test = {}
for x in df2_test['Perfume Name'].unique():
    mask = df2_test['Perfume Name']==x
    dic_test[x] = df2_test['Note'][mask]
    
new_df_test = 


#%% DEEP LEARNING
lst = []
for x in df.columns[2:22]:
    for y in df[x]:
        lst.append(y)

ingred_set = list(set(lst))

#%%

for x in range(len(ingred_set)):
    if ingred_set[x]==np.nan:
        ingred_set[x] = "nothing"
    ingred_set[x]=ingred_set[x].replace('  ', ' ').replace(' ', '_').replace('1', '').replace('0', '').replace('2', '').lower()
    print(ingred_set[x])

#%% Selecting notes cols and filling na with dummy string value

df_notes = df[df.columns[2:22]]
mask = df_notes.isna()

df_notes[mask] = ''

#%% getting the list of unique ingredients



ingredients=[]

for i in df_notes:
    # for j in df_notes[i]:
        df_notes[i]=df_notes[i].apply(lambda x : x.lower().
                          replace('  ', ' ').
                          replace(' ', '_').
                          replace('1', '').
                          replace('2', '').
                          replace('3', ''))
        lst = list(df_notes[i].unique())
        for k in lst:
            ingredients.append(k)


for i in df.columns[2:22]:
    lst.append(df[i])

df['all_notes'] = ''


df_notes['all_notes'] = df_notes['note1'] + " " + df_notes['note2'] + " " + df_notes['note3'] + " " + df_notes['note4'] + " " + df_notes['note5'] + " " + df_notes['note6'] + " " + df_notes['note7'] + " " + df_notes['note8'] + " " + df_notes['note9'] + " " + df_notes['note10'] + " " + df_notes['note11'] + " " + df_notes['note12'] + " " + df_notes['note13'] + " " + df_notes['note14'] + " " + df_notes['note15'] + " " + df_notes['note16'] + " " + df_notes['note17'] + " " + df_notes['note18'] + " " + df_notes['note19'] + " " + df_notes['note20']
series_allnotes = df_notes['all_notes']

df_treated = pd.concat([df[['brand', 'title']], series_allnotes], axis='columns')

#%%

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances


vectorizer = CountVectorizer()

corpus = (df_treated['all_notes'])

counts = vectorizer.fit_transform(corpus).todense()
vocab = vectorizer.vocabulary_


df_mat = pd.DataFrame(counts)
type(np.asarray(counts))

sns.heatmap(df_treated[''])

euclidean_distances(np.asarray(counts[0]), np.asarray(counts[1]))
euclidean_distances(np.asarray(counts[0]), np.asarray(counts[2]))
euclidean_distances(np.asarray(counts[0]), np.asarray(counts[3]))
euclidean_distances(np.asarray(counts[1]), np.asarray(counts[2]))



#%%

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt


cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))


X = np.hstack((cluster1, cluster2)).T
X = np.vstack((x, y)).T


K = range(1, 10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    
kmeans.fit(X)

meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
plt.show()

#%%

len(df[df['gender']=="nan"])/len(df)*100

