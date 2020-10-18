#!/usr/bin/python3

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from variables import input_dir, output_dir

################### Parametres simulation ##############################
remove_col_kmeans_movies = ['id','title','vote_average', 'vote_count']
trunc_user_high = 800 #nombre max de vues total par user
trunc_user_low = 20 #nombre min de vues total par user
trunc_movie_low = 1000
trunc_movie_high = 100000
kmeans_centroid_movies = 5
kmeans_centroid_users = 4
n = 5 #nombre de films à recommander

p_c_a = True # Activer ou pas la Principal Component analysis
acp_dim = 20 # Principal Component analysis

###### Modelisation Kmeans utlisateur #########
a=1
# valeur de a doit être comprise entre zero et un!!!
# Pour a = 0,
# Si l'utilisateur a offert un rating de 5 a un des films. score vaut 5
# Si l'utilisateur n'a offert un rating de 5 a aucun un des films. score vaut le score moyen qu'il a offert aux films
# Pour a =1,
# Le score vaut toujours le score moyen qu'il a offert aux films
#plus a augmente et se rapproche de 1, plus l'ecart entre avoir son film préfére dans un cluster ou pas diminue
b = 0.99 #définir un score de film pour chaque cluster d'utilisateur  : b*moyenne_note + (1-b)*part

##########################################################################




################### fichier input et output ###############################
input_dir = input_dir
output_dir = output_dir
################################################################










################ Lecture et tri de la donnée ########################
tableau_movies_full = pd.read_csv(input_dir + "final_data_movie.csv")
ratings = pd.read_csv(input_dir + "clean_ratings.csv")
ratings  = ratings.drop(['timestamp'], axis = 1)

all_movies = list(tableau_movies_full.drop_duplicates("id")["id"])
ratings = ratings[ratings["movieId"].isin(all_movies)]

nbr_votes_movie = ratings.groupby("movieId")["movieId"].count().reset_index(name= "count_movie")
ratings = pd.merge(ratings, nbr_votes_movie, left_on="movieId", right_on='movieId', how='inner')
ratings = ratings[ratings["count_movie"]>trunc_movie_low]
ratings = ratings[ratings["count_movie"]<trunc_movie_high]
ratings = ratings.drop("count_movie", axis= 1)

#legit_movies = list(ratings.drop_duplicates("movieId")["movieId"])
#tableau_movies = tableau_movies_full[tableau_movies_full["id"].isin(legit_movies)]

del nbr_votes_movie
#del legit_movies
del all_movies

tableau_movies = tableau_movies_full.drop(tableau_movies_full[remove_col_kmeans_movies], axis = 1)

#on filtre les utilisateurs qui ont emis trop de votes, ou pas assez dans ratings
data_user_votes = ratings.groupby(["userId"])["rating"].apply(lambda x : len(list(x) )).reset_index(name = 'voteCount')
data_user_votes = data_user_votes[ trunc_user_low  < data_user_votes['voteCount'] ]
data_user_votes = data_user_votes[  data_user_votes['voteCount'] < trunc_user_high]
df = data_user_votes.sort_values(by=['voteCount'])

ratings = ratings[np.isin(ratings['userId'], data_user_votes['userId'])]
ratings["title"] = pd.merge(ratings, tableau_movies_full[["id", "title"]], left_on = "movieId", right_on="id")["title"]

del data_user_votes
del df
######################################################################





####################### Principle Component Analysis #####################################

if p_c_a:
    pca = PCA(n_components=acp_dim, random_state=80)
    pca.fit(tableau_movies)
    tableau_movies = pd.DataFrame(pca.transform(tableau_movies))

####################### Kmeans sur le récapiptulatif de films #############################





####################### Kmeans sur le récapiptulatif de films #############################

#### on lance kmeans avec k clusters défini en parametre de simulation
kmeans = KMeans(n_clusters=kmeans_centroid_movies).fit(tableau_movies)
centroids = kmeans.cluster_centers_

movies = pd.DataFrame({'id': tableau_movies_full['id'], 'Kmeans_movies_cluster': kmeans.labels_})

# On ajoute le clustering à la table ratings et à la table tableau_movies_full
ratings = pd.merge(ratings, movies, left_on = "movieId", right_on = "id")
tableau_movies_full = pd.merge(tableau_movies_full, movies, left_on = "id", right_on = "id")
#########################################################################################



################ Traitement pour le kmeans users ##############################################

#permet d'avoir une info sur le cluster de film en complément du tableau ratings
user_movies = ratings.groupby(['userId', 'Kmeans_movies_cluster'])
df_score = user_movies['rating'].apply(lambda x : (1-a)*float(5) + a*sum(list(x))/len(list(x)) if float(5) in list(x) else sum(list(x))/len(list(x))).reset_index(name = 'score')
df_users = df_score.pivot(index = 'userId', columns = 'Kmeans_movies_cluster').reset_index()
df_users = df_users.replace(np.nan, 3.5)

names = []
for i in range(kmeans_centroid_movies):
    name = "k"+str(i)
    names.append(name)
df_users.columns= ["userId"] + names 

#On calcule la moyenne des note spar utilisateur afin de ne pas se retrouver avec un clustering de type :
# groupe des utilisteurs qui notent bien, groupe des utilisateurs qui notent mal ,ect

moy_by_user = ratings.groupby("userId")["rating"].mean().reset_index(name="note_moy_user")
df_users = pd.merge(df_users, moy_by_user, left_on = "userId", right_on="userId")
data = df_users
data = data.drop("userId", axis=1)
for col in data.drop("note_moy_user", axis=1).columns:
    data[col] = data[col]/data["note_moy_user"]
data = data.drop("note_moy_user", axis=1)


df_kmeans_users = data

del moy_by_user
del df_score
del data
del names
#################################################################################################






############################## Kmeans utilisateurs #######################################
kmeans = KMeans(n_clusters=kmeans_centroid_users).fit(df_kmeans_users)
centroids = kmeans.cluster_centers_

user_clusters = pd.DataFrame({'userId': df_users["userId"], 'Kmeans_user_cluster': kmeans.labels_})
# On ajoute le clustering à la tale ratings
ratings = pd.merge(ratings, user_clusters, left_on="userId", right_on="userId")

# nombre d'utilisateurs par cluster
nb_users_cluster = ratings.drop_duplicates("userId").groupby('Kmeans_user_cluster')["rating"].count().reset_index()


del user_clusters
del df_kmeans_users
del df_users
del movies
###########################################################################################





############################### Stats descriptives ############################################

### moyenne et compte de chaque duo film/cluster user
links = ratings.groupby(["Kmeans_user_cluster","movieId"])["rating"].count().reset_index(name="count")
moyenne = ratings.groupby(["Kmeans_user_cluster","movieId"])["rating"].mean().reset_index(name="mean")
variance = ratings.groupby(["Kmeans_user_cluster","movieId"])["rating"].var().reset_index(name="variance")
links["mean"] = moyenne["mean"]
links["variance"] = variance["variance"]

del moyenne
del variance

### récupérer les meilleurs films par cluster user selon la mean
parmi_combien = 300
best_movies_per_cluster = links.sort_values(["Kmeans_user_cluster",'mean'],ascending=False).groupby("Kmeans_user_cluster").head(parmi_combien).reset_index(drop=True)
best_movies_per_cluster["nb_user_cluster"] = pd.merge(best_movies_per_cluster, nb_users_cluster, left_on="Kmeans_user_cluster", right_on = "Kmeans_user_cluster")["rating"]
best_movies_per_cluster["part"] = (best_movies_per_cluster["count"] / best_movies_per_cluster["nb_user_cluster"]) * 100
best_movies_per_cluster["score"] = b*best_movies_per_cluster["mean"] + (1-b)*best_movies_per_cluster["part"]
best_movies_per_cluster = pd.merge(best_movies_per_cluster, tableau_movies_full, left_on = "movieId", right_on = "id")[["title", "Kmeans_user_cluster","Kmeans_movies_cluster", "mean", "part" ,"count", "variance"]].sort_values(["Kmeans_user_cluster", "mean"], ascending = False).reset_index()

# Lien cluster movies, cluster user
contingence_clusteruser_clustermovie = best_movies_per_cluster.groupby(["Kmeans_user_cluster", "Kmeans_movies_cluster"])["mean"].count().reset_index(name="count_per_cluster")
contingence_clusteruser_clustermovie = contingence_clusteruser_clustermovie.pivot(index = "Kmeans_user_cluster", columns = "Kmeans_movies_cluster", values = "count_per_cluster")

# visualisation des meilleurs films par cluster
visualisation_meilleurs_film_par_cluster = best_movies_per_cluster.pivot( columns = 'Kmeans_user_cluster', values ="title" )

for i in range(kmeans_centroid_users):
    index = (kmeans_centroid_users - 1 - i) *parmi_combien
    values = visualisation_meilleurs_film_par_cluster.loc[index:index+parmi_combien-1,i].reset_index(drop=True)
    visualisation_meilleurs_film_par_cluster[i] = values
    
visualisation_meilleurs_film_par_cluster = visualisation_meilleurs_film_par_cluster[0:parmi_combien-1]
####################################################################################################




############################ Recommendations pour chaque utilisateur ###############################
# On veut maintenant recommander n films  à chaque utilisateur
# On commence par le meilleur film selon son groupe et on descend jusqu'à qu'il y ait n films à lui recommander

def delete_if_in_other(row):
    return [x for x in row["recommended"] if x not in row["title"]][0:n]

def recommendations():
    for i in range(kmeans_centroid_users):
        films = list(best_movies_per_cluster[best_movies_per_cluster["Kmeans_user_cluster"]==i]["title"])
        users = ratings[ratings["Kmeans_user_cluster"]==i]
        user_movie_list = users.groupby("userId")["title"].apply(list).reset_index()
        user_movie_list["recommended"] = [films for _ in range(len(user_movie_list))]
        user_movie_list["recommended"] = user_movie_list.apply(delete_if_in_other, axis=1)
        user_movie_list = user_movie_list.drop("title", axis=1)
        
        if i == 0:
            toutes_les_recommendations = user_movie_list
        else:
            toutes_les_recommendations = pd.concat([toutes_les_recommendations, user_movie_list])
    
    del user_movie_list
    del films
    del users
    return toutes_les_recommendations

recommendations = recommendations()




# Sauvegarde des recommendations
recommendations.to_csv(output_dir + "recommendations.csv", index= False)











































#Graphs (à mettre en commentaires)

#import ast
#
#a = pd.read_csv("/home/fitec/donnees_films/metadata_carac_speciaux.csv")
#
#a=a.dropna(subset=['id'])
#a=a.dropna(subset=['title'])
#a=a.loc[a['status']== 'Released']
#a=pd.get_dummies(a, columns=["adult"]) 
#a=a.drop_duplicates()
#a=a.drop_duplicates(subset='id', keep="first")
#a=a.drop_duplicates(subset='title', keep="first")
#a=a.reset_index(drop=True)
#
#def encoding_dic(data, variable, liste):
#
#    serie_col = data[variable]
#    #Création de la colonne total : liste des catégories appartenant à la liste pour chaque ligne
#    def add(x, liste_col):
#        total = []
#        if type(x) == str and x[0] == "[":
#            a = ast.literal_eval(x)
#            if len(a) > 0:
#                for j in range(len(a)):
#                    comp = a[j]["name"]
#                    if comp in liste_col:
#                        total.append(comp)
#                if len(total) == 0:
#                    total.append("autre")
#            else:
#                total.append("autre")
#        return total
#    
#    total = serie_col.apply(lambda x : add(x, liste_col = liste))
#    df = serie_col.to_frame()
#    df["total"] = total
#    return df
#
#liste_genre = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Documentary']
#liste_prod_comp = ['WarnerBros.', 'Metro-Goldwyn-MayerMGM', 'ParamountPictures', 'TwentiethCenturyFoxFilmCorporation', 'UniversalPictures', 'ColumbiaPicturesCorporation', 'Canal', 'ColumbiaPictures', 'RKORadioPictures']
#liste_prod_count = ['UnitedStatesofAmerica', 'null', 'UnitedKingdom', 'France', 'Germany', 'Italy', 'Canada', 'Japan', 'Spain', 'Russia']
#
#
#genres = encoding_dic(a, "genres", liste_genre)
#genres["genre"] = genres["total"].apply(lambda x :  x[0])
#prod_comp = encoding_dic(a, "production_companies", liste_prod_comp)
#prod_comp["prod_comp"] = prod_comp["total"].apply(lambda x :  x[0])
#prod_count = encoding_dic(a, "production_countries", liste_prod_count)
#prod_count["prod_count"] = prod_count["total"].apply(lambda x :  x[0])
#
#
#
#b = pd.concat([a["title"], genres["genre"], prod_comp["prod_comp"], prod_count["prod_count"]], axis = 1)
#
#c = pd.merge(best_movies_per_cluster, b, left_on = "title", right_on = "title").sort_values(["Kmeans_user_cluster", "mean"], ascending=False)
#
#
##Save for graphs
#c.to_csv("/home/fitec/donnees_films/for_graphs/best_movies_per_cluster.csv", index= False)


