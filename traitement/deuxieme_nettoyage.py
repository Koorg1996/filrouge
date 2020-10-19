import pandas as pd
import numpy as np
import matplotlib as plt
import ast
from ast import literal_eval
from variables import input_dir, output_dir


path = input_dir

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
metadata=metadata.drop_duplicates().sort_values("popularity", ascending=False)
metadata=metadata.drop_duplicates(subset='id', keep="first")
metadata=metadata.drop_duplicates(subset='title', keep="first")
metadata = metadata.reset_index(drop=True)







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
                    total.append("none")
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
liste_genre = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Documentary']
liste_prod_comp = ['WarnerBros.', 'Metro-Goldwyn-MayerMGM', 'ParamountPictures', 'TwentiethCenturyFoxFilmCorporation', 'UniversalPictures', 'ColumbiaPicturesCorporation', 'Canal', 'ColumbiaPictures', 'RKORadioPictures']
liste_prod_count = ['UnitedStatesofAmerica', 'null', 'UnitedKingdom', 'France', 'Germany', 'Italy', 'Canada', 'Japan', 'Spain', 'Russia']

dfgenres = encoding_dic(data=metadata, variable="genres", liste=liste_genre)
dfgenres=dfgenres.drop(['genres', 'total'], axis=1)
dfprodcomp = encoding_dic(data=metadata, variable="production_companies", liste=liste_prod_comp)
dfprodcomp=dfprodcomp.drop(['production_companies', 'total'], axis=1)
dfprodcount = encoding_dic(data=metadata, variable="production_countries", liste=liste_prod_count)
dfprodcount=dfprodcount.drop(['production_countries', 'total'], axis=1)

datamovienew=pd.concat([metadata, dfgenres, dfprodcomp, dfprodcount], axis=1)

datamovienew=datamovienew.drop(['genres', 'production_companies', 'production_countries'], axis=1)

#                                   Fin du travail sur les variables dictionnaire















#7 ajouter la variable keywords de la table keywords
keywords=keywords.dropna(subset=['id'])
keywords=keywords.drop_duplicates(subset='id', keep="first")

liste_key=['woman director', 'independent film', 'murder', 'based on novel', 'musical', 'sex', 'violence', 'nudity', 'biography', 'revenge', 'suspense', 'love', 'female nudity', 'sport', 'police', 'teenager', 'duringcreditsstinger', 'sequel', 'friendship', 'world war ii', 'drug', 'prison', 'stand-up comedy', 'high school', 'martial arts', 'suicide', 'kidnapping', 'rape', 'silent film', 'film noir', 'family', 'serial killer', 'monster', 'alien', 'dystopia', 'paris', 'new york', 'blood', 'gay', 'short', 'marriage', 'christmas', 'gore', 'zombie', 'death', 'gangster', 'small town', 'london england', 'romance', 'prostitute', 'detective', 'aftercreditsstinger', 'male nudity', 'robbery', 'vampire', 'father son relationship', 'wedding', 'los angeles', 'escape', 'dog', 'teacher', 'holiday', 'war', 'magic', 'hospital', 'doctor', 'music', 'remake', 'jealousy', 'based on true story', 'ghost', 'party', 'island', 'spy', 'new york city', 'lgbt', 'japan', 'daughter', 'investigation', 'coming of age', 'money', 'superhero', 'infidelity', 'corruption', 'torture', 'brother brother relationship', 'homosexuality', 'nazis', 'adultery', 'extramarital affair', 'wife husband relationship', 'slasher', 'supernatural', 'lawyer', 'dark comedy', 'friends', 'scientist']
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
    if len(a.loc[i,'release_date']) < 4:
        var.append(a0)
    elif (len(a.loc[i,'release_date']) >= 4 and int(a.loc[i,'release_date']) <= 1990) :
        var.append(a1)
    elif (len(a.loc[i,'release_date']) >= 4 and 1990 < int(a.loc[i,'release_date']) <= 2010) :
        var.append(a2)
    elif (len(a.loc[i,'release_date']) >= 4 and int(a.loc[i,'release_date']) > 2010):
        var.append(a3)
    else:
        var.append(a0)

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





