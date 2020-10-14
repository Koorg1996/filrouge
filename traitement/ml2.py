#!/usr/bin/python3

import pandas as pd
import numpy as np
import os
#import json
#import ast
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
kmeans_centroid_users = 6

date_time = datetime.datetime.now()
################### fichier output ##############################
output_dir = "data_csv/more/output_"+str(date_time) +'/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
content_file =  'Colonnes retirées de tableau_movies'  + str(remove_col_kmeans_movies) +'\n'
#content_file.append
f = open(output_dir+"output"+str(date_time)+".txt", "a")
f.write(content_file)
f.close()
############################## ##############################

################### fichier input ###############################
input_dir = "/home/fitec/donnees_films/"
################################################################


#Lecture et tri de la donnée
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







data_user_votes = ratings.groupby(["userId"])["rating"].apply(lambda x : len(list(x) )).reset_index(name = 'voteCount')
data_user_votes = data_user_votes[ trunc_user_low  < data_user_votes['voteCount'] ]
data_user_votes = data_user_votes[  data_user_votes['voteCount'] < trunc_user_high]
df = data_user_votes.sort_values(by=['voteCount'])
fig = px.scatter(x=pd.Series(range(0,len(df['userId']))), y=df['voteCount'])
fig.write_html(output_dir + "Repartition du nombre de vues par utilisateur.html")

#on filtre les utilisateurs qui ont emis trop de votes, ou pas assez
ratings = ratings[np.isin(ratings['userId'], data_user_votes['userId'])]


#On observe le plus haut rating qu'un utlisateur a offert aux films qu'il a noté
maxrating = ratings.groupby(["userId"])["rating"].apply(lambda x : max(x)).reset_index()




# Kmeans sur le récapiptulatif de films

##### Critère de Coude pour Kmeans movies


Inertie =[]
n_centroids = coude_centroid_movies
for i in range(1, n_centroids):
    kmeans = KMeans(n_clusters = i).fit(tableau_movies)
    Inertie.append(kmeans.inertia_)

plt.plot(range(1, n_centroids), Inertie)
plt.title('Critere de Coude Kmeans movies')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.show()

#### choix du nombre de clusters pour le Kmeans movies
kmeans = KMeans(n_clusters=kmeans_centroid_movies).fit(tableau_movies)
centroids = kmeans.cluster_centers_

movies = pd.DataFrame({'id': tableau_movies_full['id'], 'Kmeans_movies_cluster': kmeans.labels_})
#movies

fig = px.histogram(movies, x="Kmeans_movies_cluster", title = "Repartition des films par cluster Kmeans movies ")
fig.update_xaxes(type='category')
#fig.show()
fig.write_html(output_dir + "Repartition des films par cluster Kmeans movies.html")





fig = px.histogram(maxrating, x="rating", title = "Repartition du plus haut score offert par un utilisateur")
fig.update_xaxes(type='category')
#fig.show()
fig.write_html(output_dir + "Repartition du plus haut score offert par un utilisateur.html")


#On observe combien de film ont noté les utilisateurs ayant offert un score maximal assez bas
df = data_user_votes[np.isin(data_user_votes['userId'], maxrating[maxrating['rating']<4]['userId'])]
df = df.sort_values(by = ['voteCount'])
fig = px.scatter(x=pd.Series(range(0,len(df['userId']))), y=df['voteCount'])
#fig.show()
fig.write_html(output_dir + "Repartition du nombre de vues par utilisateurayant offert un score maximal assez bas.html")



#permet d'avoir une info sur le cluster de film en complément du tableau ratings
ratings_merge = pd.merge(ratings, movies, left_on ='movieId', right_on = 'id') 

user_movies = ratings_merge.groupby(['userId', 'Kmeans_movies_cluster'])

df_score = user_movies['rating'].apply(lambda x : float(5) if float(5) in list(x) else 0).reset_index(name = 'score')
#df_score

df_users = df_score.pivot(index = 'userId', columns = 'Kmeans_movies_cluster').reset_index()
df_users = df_users.replace(np.nan, 0)

df_kmeans_users = df_users['score']


#Kmeans utilisateurs

## Critère de Coude
Inertie =[]
n_centroids = coude_centroid_users
for i in range(1, n_centroids):
    kmeans = KMeans(n_clusters = i).fit(df_kmeans_users)
    #kmeans.fit(x)
    Inertie.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, n_centroids), Inertie)
plt.title('Critere de Coude Kmeans utilisateurs')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.show()


#### choix du nombre de clusters pour le Kmeans utilisateurs
kmeans = KMeans(n_clusters=kmeans_centroid_users).fit(df_kmeans_users)
centroids = kmeans.cluster_centers_

user_clusters = pd.DataFrame({'userId': df_users[('userId', '')], 'Kmeans_user_cluster': kmeans.labels_})


fig = px.histogram(user_clusters, x="Kmeans_user_cluster", title = "Repartition des utilisateurs par cluster utilisateurs")
fig.update_xaxes(type='category')
#fig.show()
fig.write_html(output_dir + "Repartition des utilisateurs par cluster utilisateurs.html")




date_timeend = datetime.datetime.now()
runtime = date_timeend - date_time
#Maintenant, on souhaite retrouver la liste de films vues par un groupe d'utilisateurs


