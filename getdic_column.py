import pandas as pd
import numpy as np
import json
import ast


fichier = "movies_metadata.csv"
link = "D:/filrouge/data/"+fichier
data = pd.read_csv(link)

def encoding_dic(table, variable, liste):
    
    #Colonne concernée
    #serie_col = data["genres"]
        
    #Liste des catégories conservées
    #liste = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Documentary', 'null']
    
    serie_col = data[variable]
    #Création de la colonne total : liste des catégories appartenant à la liste pour chaque ligne
    def add(x, liste_col):
        total = []
        if type(x) == str and x[0] == "[":
            a = ast.literal_eval(x)
            if len(a) > 0:
                for j in range(len(a)):
                    comp = a[j]["name"]
                    if comp in liste_col:
                        total.append(comp)
                if len(total) == 0:
                    total.append("null")
            else:
                total.append("null")
        return total
    
    total = serie_col.apply(lambda x : add(x, liste_col = liste))
    df = serie_col.to_frame()
    df["total"] = total
    
    #Création des colonnes pour le OneHotEncoding
    for genre in liste:
        df[genre] = 0
    
    #Complétion des colonnes OneHotEncoding grâce à la colonne total
    def add2(x,genre_cherche):
        for genre in x["total"]:
            if genre == genre_cherche:
                return 1
        return 0
    
    for genre in liste:
        df[genre] = df.apply(lambda x : add2(x, genre_cherche = genre), axis=1)
    
    return df


liste_cat = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Documentary', 'null']
    
df = encoding_dic(table=data, variable="genres", liste=liste_cat)




















