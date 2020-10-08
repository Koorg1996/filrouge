import pandas as pd
import numpy as np
import matplotlib as plt
import ast
from ast import literal_eval

path = "/home/fitec/donnees_films/"

#IMPORT DATA
metadata = pd.read_csv(path + "metadata_carac_speciaux.csv")
keywords=pd.read_csv(path + "keywords_carac_speciaux.csv", delimiter = ',')
ratings=pd.read_csv(path + "ratings.csv", delimiter = ',')



#                                           NETTOYAGE



#1 supprimer les valeurs missing
metadata=metadata.dropna(subset=['id'])
metadata=metadata.dropna(subset=['title'])






#2 selectionner les film=released
metadata=metadata.loc[metadata['status']== 'Released']






#3 encode adult var
metadata=pd.get_dummies(metadata, columns=["adult"])








#4 drop duplicates  
metadata=metadata.drop_duplicates()
metadata=metadata.drop_duplicates(subset='id', keep="first")








#5 selection variables
metadata=metadata[['genres', 
                     'id',
                     'original_language', 
                     'production_companies', 
                     'production_countries', 
                     'release_date',
                     'title'  , 
                     'vote_average',
                     'vote_count',
                     'adult_False', 
                     'adult_True'      ]]














#6                                           Travail sur les variables dictionnaire

###############################fonction pr obtenir une liste des categories (max 20 elements a priori) a partir des dictionnaires
def categorie (data, variable):
    data['genrestest'] = data[variable].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    liste=data['genrestest'].str[0].value_counts()[:20].index.tolist()
    return liste
#######################################################

liste_genre=categorie(metadata, 'genres')
liste_pcomp=categorie(metadata, 'production_companies')
liste_pcount=categorie(metadata, 'production_countries')

metadata=metadata.drop(['genrestest'], axis=1)

#########################fonction encoding pour attribuer 1 a chaque element qui existe dans la liste des categories############################
def encoding_dic(data, variable, liste):

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

##########################################################

dfgenres = encoding_dic(data=metadata, variable="genres", liste=liste_genre)
dfgenres=dfgenres.drop(['genres', 'total'], axis=1)
dfprodcomp = encoding_dic(data=metadata, variable="production_companies", liste=['null', 'Warner Bros.', 'Metro-Goldwyn-Mayer (MGM)', 'Paramount Pictures'])
dfprodcomp=dfprodcomp.drop(['production_companies', 'total'], axis=1)
dfprodcount = encoding_dic(data=metadata, variable="production_countries", liste=liste_pcount)
dfprodcount=dfprodcount.drop(['production_countries', 'total'], axis=1)

datamovienew=pd.concat([metadata, dfgenres, dfprodcomp, dfprodcount], axis=1)

datamovienew=datamovienew.drop(['genres', 'production_companies', 'production_countries'], axis=1)

#                                   Fin du travail sur les variables dictionnaire















#7 ajouter la variable keywords de la table keywords
keywords=keywords.dropna(subset=['id'])
keywords=keywords.drop_duplicates(subset='id', keep="first")

liste_key=categorie(keywords, 'keywords')
keywords=keywords.drop(['genrestest'], axis=1)

dfkey= encoding_dic(data=keywords, variable="keywords", liste=liste_key)
dfkey=dfkey.drop(['keywords', 'total'], axis=1)

datakey=pd.concat([keywords, dfkey], axis=1)
datakey=datakey.drop(['keywords'], axis=1)

datamovienew=pd.merge(datamovienew,datakey, on='id')










#8 Catégorisation de la variable release_date
var = []
a0 = "date inconnue"
a1 = "films anciens"
a2 = "films récents"
a3 = "films très récents"

dates = datamovienew["release_date"]
a = dates.apply(lambda x : str(x))
a = pd.DataFrame(a.apply(lambda x : x[0:4]))

for i in range(0,len(a)):
    if (len(a.loc[i,'release_date']) < 4 ) :
        var.append(a0)
    elif (len(a.loc[i,'release_date']) >= 4 and int(a.loc[i,'release_date']) <= 1990) :
        var.append(a1)
    elif (len(a.loc[i,'release_date']) >= 4 and 1990 < int(a.loc[i,'release_date']) <= 2010) :
        var.append(a2)
    elif (len(a.loc[i,'release_date']) >= 4 and int(a.loc[i,'release_date']) > 2010):
        var.append(a3)

datamovienew["dates_types"] = var
datamovienew=pd.get_dummies(datamovienew, columns=["dates_types"])
datamovienew=datamovienew.drop(['release_date'], axis=1)













#9 Catégorisation de la variable original language
datamovienew["original_language"].unique()

def only_these_languages(x):
    if x not in ["fr", "en", "it", "ja", "de"]:
        return "other"
    else:
        return x
    
datamovienew["original_language"] = datamovienew["original_language"].apply(lambda x : only_these_languages(x))  
datamovienew=pd.get_dummies(datamovienew, columns=["original_language"])
final_data_movie = datamovienew



#                                  Fin du nettoyage


               
#10 On save la table 
final_data_movie.to_csv(path + "final_data_movie.csv", index= False)





