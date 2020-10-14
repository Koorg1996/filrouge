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








<<<<<<< HEAD:traitement/ml3.py

=======
######################## FIGURE 1 ##############################
fig1 = px.scatter(x=pd.Series(range(0,len(df['userId']))), y=df['voteCount'], title = "Nombre de vues total par utilisateur selectionné")
fig1.write_html(output_dir + "Nombre de vues total par utilisateur.html")
>>>>>>> 820f76d14e14e436984be3a749e98bf683cd1045:traitement/ml2.py






######################## Differents graphs pour illustrer ##############################
#On observe le plus haut rating qu'un utlisateur a offert aux films qu'il a noté
maxrating = ratings.groupby(["userId"])["rating"].apply(lambda x : max(x)).reset_index()

fig3 = px.histogram(maxrating, x="rating", title = "Repartition du plus haut score offert par un utilisateur")
fig3.update_xaxes(type='category')

#On observe combien de film ont noté les utilisateurs ayant offert un score maximal assez bas
df = data_user_votes[np.isin(data_user_votes['userId'], maxrating[maxrating['rating']<4]['userId'])]
df = df.sort_values(by = ['voteCount'])
fig4 = px.scatter(x=pd.Series(range(0,len(df['userId']))), y=df['voteCount'])



fig1 = px.scatter(x=pd.Series(range(0,len(df['userId']))), y=df['voteCount'])

<<<<<<< HEAD:traitement/ml3.py
del maxrating
del data_user_votes
#####################################################################################










####################### Kmeans sur le récapiptulatif de films #############################

##### Critère de Coude pour Kmeans movies

=======
######################## COUDE KMEANS MOVIES ##############################
Inertie =[]
n_centroids = coude_centroid_movies
for i in range(1, n_centroids):
    kmeans = KMeans(n_clusters = i).fit(tableau_movies)
    Inertie.append(kmeans.inertia_)
>>>>>>> 820f76d14e14e436984be3a749e98bf683cd1045:traitement/ml2.py

#Inertie =[]
#n_centroids = coude_centroid_movies
#for i in range(1, n_centroids):
#    kmeans = KMeans(n_clusters = i).fit(tableau_movies)
#    Inertie.append(kmeans.inertia_)

<<<<<<< HEAD:traitement/ml3.py
####### COUDE KMEANS MOVIES #######
#plt.plot(range(1, n_centroids), Inertie)
#plt.title('Critere de Coude Kmeans movies')
#plt.xlabel('Nombre de clusters')
#plt.ylabel('Inertie')
#plt.show()
=======

coude_movies = plt.figure()
plt.plot(range(1, n_centroids), Inertie)
plt.title('Critere de Coude Kmeans movies')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
coude_movies.show()
>>>>>>> 820f76d14e14e436984be3a749e98bf683cd1045:traitement/ml2.py

#### on lance kmeans avec k clusters défini en parametre de simulation
kmeans = KMeans(n_clusters=kmeans_centroid_movies).fit(tableau_movies)
centroids = kmeans.cluster_centers_

movies = pd.DataFrame({'id': tableau_movies_full['id'], 'Kmeans_movies_cluster': kmeans.labels_})

# On ajoute le clustering à la table ratings
ratings = pd.merge(ratings, movies, left_on = "movieId", right_on = "id")

###### Graph de la répartition par cluster de film ######
fig2 = px.histogram(movies, x="Kmeans_movies_cluster", title = "Repartition des films par cluster Kmeans movies ")
fig2.update_xaxes(type='category')
<<<<<<< HEAD:traitement/ml3.py

#########################################################################################




=======
fig2.write_html(output_dir + "Repartition des films par cluster Kmeans movies.html")



######################## FIGURE 3 ##############################
fig3 = px.histogram(maxrating, x="rating", title = "Repartition du plus haut score offert par un utilisateur")
fig3.update_xaxes(type='category')
fig3.write_html(output_dir + "Repartition du plus haut score offert par un utilisateur.html")
>>>>>>> 820f76d14e14e436984be3a749e98bf683cd1045:traitement/ml2.py


<<<<<<< HEAD:traitement/ml3.py

=======
######################## FIGURE  4 ##############################
fig4 = px.scatter(x=pd.Series(range(0,len(df['userId']))), y=df['voteCount'],title = "Nombre de films vues par utilisateur ayant offert un score maximal bas")
#fig4.write_html(output_dir + "Nombre de films vues par utilisateur ayant offert un score maximal bas.html")
del data_user_votes
>>>>>>> 820f76d14e14e436984be3a749e98bf683cd1045:traitement/ml2.py



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

## Critère de Coude
<<<<<<< HEAD:traitement/ml3.py
#Inertie =[]
#n_centroids = coude_centroid_users
#for i in range(1, n_centroids):
#    kmeans = KMeans(n_clusters = i).fit(df_kmeans_users)
#    Inertie.append(kmeans.inertia_)

######################## Coude users ##############################
#plt.plot(range(1, n_centroids), Inertie)
#plt.title('Critere de Coude Kmeans utilisateurs')
#plt.xlabel('Nombre de clusters')
#plt.ylabel('Inertie')
#plt.show()
=======
######################## Coude users ##############################
Inertie =[]
n_centroids = coude_centroid_users
for i in range(1, n_centroids):
    kmeans = KMeans(n_clusters = i).fit(df_kmeans_users)
    #kmeans.fit(x)
    Inertie.append(kmeans.inertia_)


coude_users = plt.figure()
plt.plot(range(1, n_centroids), Inertie)
plt.title('Critere de Coude Kmeans utilisateurs')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
coude_users.show()
>>>>>>> 820f76d14e14e436984be3a749e98bf683cd1045:traitement/ml2.py


#### choix du nombre de clusters pour le Kmeans utilisateurs
kmeans = KMeans(n_clusters=kmeans_centroid_users).fit(df_kmeans_users)
centroids = kmeans.cluster_centers_

user_clusters = pd.DataFrame({'userId': df_users[('userId', '')], 'Kmeans_user_cluster': kmeans.labels_})
# On ajoute le clustering à la tale ratings
ratings = pd.merge(ratings, user_clusters, left_on="userId", right_on="userId")

###### Graph de la répartition par cluster des users ######
fig5 = px.histogram(user_clusters, x="Kmeans_user_cluster", title = "Repartition des utilisateurs par cluster utilisateurs")
fig5.update_xaxes(type='category')
<<<<<<< HEAD:traitement/ml3.py

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
best_movies_per_cluster = links.sort_values(["Kmeans_user_cluster",'mean'],ascending=False).groupby("Kmeans_user_cluster").head(100)
best_movies_per_cluster = pd.merge(best_movies_per_cluster, tableau_movies_full, left_on = "movieId", right_on = "id")[["title", "Kmeans_user_cluster","Kmeans_movies_cluster", "mean", "count"]].sort_values(["cluster_user", "mean"])


=======
fig5.write_html(output_dir + "Repartition des utilisateurs par cluster utilisateurs.html")
>>>>>>> 820f76d14e14e436984be3a749e98bf683cd1045:traitement/ml2.py

w = best_movies_per_cluster.groupby(["cluster_user", "cluster_movie"])["mean"].count().reset_index(name="count_per_cluster")

#Maintenant, on souhaite retrouver la liste de films vues par un groupe d'utilisateurs


