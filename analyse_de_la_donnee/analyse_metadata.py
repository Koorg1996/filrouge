import pandas as pd
import numpy as np
import json
import ast


fichier = "movies_metadata.csv"
link = "D:/filrouge/data/"+fichier
data = pd.read_csv(link)

columns_to_drop = ["homepage", "imdb_id", "overview", "poster_path", "status", "tagline", "title"]
df = data.drop(columns_to_drop, axis=1)




# Statistiques des films triés par une colonne de format dictionnaire, pour l'instant par pays

def find_name(LIGNE, col, name):
    ligne = LIGNE[col]
    if type(ligne) == str and ligne[0] == '[':
        ev = ast.literal_eval(ligne)
        if len(ev) > 0:
            for enum in ev:
                if enum["name"] != name:
                    LIGNE["todrop"] = 1
        else:
            LIGNE["todrop"] = 1
    else:
        LIGNE["todrop"] = 1
    return LIGNE

def most_per_country():
    list_of_countries = ['United States of America', 'null', 'United Kingdom', 'France', 'Germany', 'Italy', 'Canada', 'Japan', 'Spain', 'Russia', 'India', 'Hong Kong', 'Sweden', 'Australia']
    print(list_of_countries)
    country = input("pick a country")
    if country not in list_of_countries:
        return
    
    aaa = df.copy()
    aaa["todrop"] = 0               
    aaa = aaa.apply(lambda x : find_name(x, col="production_countries", name=country), axis=1)
    
    region_movies = aaa[aaa["todrop"] == 0][["original_title", "vote_average", "vote_count"]].sort_values("vote_average", ascending= False)
    result = region_movies[region_movies["vote_count"] > 200]
    return result

# Caractérisation des catégorie rangées dans des colonnes dictionnaire

def add(x, total):
    items = []
    if type(x) == str and x[0] == "[":
        a = ast.literal_eval(x)
        if len(a) > 0:
            for j in range(len(a)):
                comp = a[j]["name"]
                items.append(comp)
        else:
            items.append("null")
    else:
        items.append("null")
    total.append(items)

def get_one_of_k(liste, first_elements):
    for item in first_elements:
        if item in liste:
            return 1   
    return 0
        
def analyse_dictio(col, max_k = 16):
    ID = df['id']
    col = df[col]
    total_obj = []
    
            
    col.apply(lambda x : add(x,total = total_obj))
    all_in_list = pd.DataFrame([i for l in total_obj for i in l], columns = ["objets"])
    count = all_in_list.groupby("objets")["objets"].count().reset_index(name="count").sort_values("count", ascending = False)
    
    k = len(count["count"])
    if k > max_k:
        k = max_k
    
    for i in range(2,k,2):
        first_k = count["objets"].tolist()[:i]
        
        result = []
        n = 0
        
        for li in total_obj:
            x = get_one_of_k(li, first_elements=first_k)
            if x == 1:
                n += 1
            result.append(x) 
            
        print("Si on prend les "+str(i)+" premiers genres "+str(n/len(total_obj)*100)+" % des films comportent au moins un de ces genres ") 
        print(first_k) 

analyse_dictio("genres")
   
             
