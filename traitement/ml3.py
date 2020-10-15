#!/usr/bin/python3

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import datetime

################### Parametres simulation ##############################
remove_col_kmeans_movies = ['id','title','vote_average', 'vote_count']
trunc_user_high = 400 #nombre max de vues total par user
trunc_user_low = 20 #nombre min de vues total par user
trunc_movie_low = 1000
trunc_movie_high = 100000
coude_centroid_movies = 20
kmeans_centroid_movies = 4
coude_centroid_users  =9
kmeans_centroid_users = 4
n = 5 #nombre de films à recommander
##########################################################################






################### fichier input ###############################
input_dir = "data_csv/"
input_dir = "/home/fitec/donnees_films/"
################################################################












################ Lecture et tri de la donnée ########################
tableau_movies_full = pd.read_csv(input_dir + "final_data_movie.csv")
ratings = pd.read_csv(input_dir + "ratings.csv")
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
######################################################################









######################## Differents graphs pour illustrer ##############################

fig1 = px.scatter(x=pd.Series(range(0,len(df['userId']))), y=df['voteCount'], title = "Nombre de vues total par utilisateur selectionné")

#On observe le plus haut rating qu'un utlisateur a offert aux films qu'il a noté
maxrating = ratings.groupby(["userId"])["rating"].apply(lambda x : max(x)).reset_index()

fig3 = px.histogram(maxrating, x="rating", title = "Repartition du plus haut score offert par un utilisateur")
fig3.update_xaxes(type='category')

#On observe combien de film ont noté les utilisateurs ayant offert un score maximal assez bas
df = data_user_votes[np.isin(data_user_votes['userId'], maxrating[maxrating['rating']<4]['userId'])]
df = df.sort_values(by = ['voteCount'])
fig4 = px.scatter(x=pd.Series(range(0,len(df['userId']))), y=df['voteCount'])

del maxrating
del data_user_votes
del df
#####################################################################################










####################### Kmeans sur le récapiptulatif de films #############################

##### Critère de Coude pour Kmeans movies


######################## COUDE KMEANS MOVIES ##############################

#Inertie =[]
#n_centroids = coude_centroid_movies
#for i in range(1, n_centroids):
#    kmeans = KMeans(n_clusters = i).fit(tableau_movies)
#    Inertie.append(kmeans.inertia_)

####### COUDE KMEANS MOVIES #######
#plt.plot(range(1, n_centroids), Inertie)
#plt.title('Critere de Coude Kmeans movies')
#plt.xlabel('Nombre de clusters')
#plt.ylabel('Inertie')
#plt.show()


#### on lance kmeans avec k clusters défini en parametre de simulation
kmeans = KMeans(n_clusters=kmeans_centroid_movies).fit(tableau_movies)
centroids = kmeans.cluster_centers_

movies = pd.DataFrame({'id': tableau_movies_full['id'], 'Kmeans_movies_cluster': kmeans.labels_})

# On ajoute le clustering à la table ratings et à la table tableau_movies_full
ratings = pd.merge(ratings, movies, left_on = "movieId", right_on = "id")
tableau_movies_full = pd.merge(tableau_movies_full, movies, left_on = "id", right_on = "id")

###### Graph de la répartition par cluster de film ######
fig2 = px.histogram(movies, x="Kmeans_movies_cluster", title = "Repartition des films par cluster Kmeans movies ")
fig2.update_xaxes(type='category')

#########################################################################################



################ Traitement pour le kmeans users ##############################################

#permet d'avoir une info sur le cluster de film en complément du tableau ratings
user_movies = ratings.groupby(['userId', 'Kmeans_movies_cluster'])
df_score = user_movies['rating'].apply(lambda x : 1 if float(5) in list(x) else 0).reset_index(name = 'score')
df_users = df_score.pivot(index = 'userId', columns = 'Kmeans_movies_cluster').reset_index()
df_users = df_users.replace(np.nan, 0)

df_kmeans_users = df_users['score']

del df_score

#################################################################################################








############################## Kmeans utilisateurs #######################################


######################## Coude users ##############################
#Inertie =[]
#n_centroids = coude_centroid_users
#for i in range(1, n_centroids):
#    kmeans = KMeans(n_clusters = i).fit(df_kmeans_users)
#    Inertie.append(kmeans.inertia_)
#
#
#coude_users = plt.figure()
#plt.plot(range(1, n_centroids), Inertie)
#plt.title('Critere de Coude Kmeans utilisateurs')
#plt.xlabel('Nombre de clusters')
#plt.ylabel('Inertie')
#coude_users.show()


#### choix du nombre de clusters pour le Kmeans utilisateurs
kmeans = KMeans(n_clusters=kmeans_centroid_users).fit(df_kmeans_users)
centroids = kmeans.cluster_centers_

user_clusters = pd.DataFrame({'userId': df_users[('userId', '')], 'Kmeans_user_cluster': kmeans.labels_})
# On ajoute le clustering à la tale ratings
ratings = pd.merge(ratings, user_clusters, left_on="userId", right_on="userId")

###### Graph de la répartition par cluster des users ######
fig5 = px.histogram(user_clusters, x="Kmeans_user_cluster", title = "Repartition des utilisateurs par cluster utilisateurs")


del user_clusters
del df_kmeans_users
del df_users
del movies
###########################################################################################





############################### Stats descriptives ############################################

### moyenne et compte de chaque duo film/cluster user
links = ratings.groupby(["Kmeans_user_cluster","movieId"])["rating"].count().reset_index(name="count")
moyenne = ratings.groupby(["Kmeans_user_cluster","movieId"])["rating"].mean().reset_index(name="mean")
links["mean"] = moyenne["mean"]

### récupérer les meilleurs films par cluster user
parmi_combien = 100
best_movies_per_cluster = links.sort_values(["Kmeans_user_cluster",'mean'],ascending=False).groupby("Kmeans_user_cluster").head(parmi_combien)
best_movies_per_cluster = pd.merge(best_movies_per_cluster, tableau_movies_full, left_on = "movieId", right_on = "id")[["title", "Kmeans_user_cluster","Kmeans_movies_cluster", "mean", "count"]].sort_values(["Kmeans_user_cluster", "mean"]).reset_index()

# Lien cluster movies, cluster user
w = best_movies_per_cluster.groupby(["Kmeans_user_cluster", "Kmeans_movies_cluster"])["mean"].count().reset_index(name="count_per_cluster")


# visualisation des meilleurs films par cluster
visualisation = best_movies_per_cluster.pivot( columns = 'Kmeans_user_cluster', values ="title" )
for i in range(kmeans_centroid_users):
    index = i*parmi_combien
    values = visualisation.loc[index:index+parmi_combien-1,i].reset_index(drop=True)
    visualisation[i] = values
visualisation = visualisation[0:parmi_combien-1]



# On veut maintenant recommander n films  à chaque utilisateur
# On commence par le meilleur film selon son groupe et on descend jusqu'à qu'il y ait n films à lui recommander

for i in range(kmeans_centroid_users):
    films = best_movies_per_cluster[best_movies_per_cluster["Kmeans_user_cluster"]==i]
    users = ratings[ratings["Kmeans_user_cluster"]==i]
    users = users[~users["movieId"].isin(films["index"])]
    recommendations_cluster = pd.merge(films, users, left_on="index", right_on="movieId").groupby("userId").head(5)
    
    if i == 0:
        toutes_les_recommendations = recommendations_cluster
    else:
        pd.concat([toutes_les_recommendations, recommendations_cluster])
    
    del users
    del recommendations_cluster

toutes_les_recommendations.groupby("userId").count()
ratings.groupby("userId").count()


films = list(best_movies_per_cluster[best_movies_per_cluster["Kmeans_user_cluster"]==0][["index"]]["index"])
users = ratings[ratings["Kmeans_user_cluster"]==0]







