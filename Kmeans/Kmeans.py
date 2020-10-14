import pandas as pd
import numpy as np
import os
#import json
#import ast
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler


################### Parametres simulation ##############################
remove_col_kmeans_movies = ['id', 'title',"vote_average", "vote_count"]
coude_centroid_movies = 9
kmeans_centroid_movies = 4
trunc_user_high = 400 #nombre max de vues total par user
trunc_user_low = 20 #nombre min de vues total par user
coude_centroid_users  =9
kmeans_centroid_users = 6
############################## ##############################


# Kmeans sur le récapiptulatif de films

tableau_movies_full = pd.read_csv('data_csv/more/final_data_movie.csv', index_col=0)
#tableau_movies_full
tableau_movies = tableau_movies_full.drop(tableau_movies_full[remove_col_kmeans_movies], axis = 1)
#centrer réduire les valeurs dans vote_count et vote_average
sc = StandardScaler()
tableau_movies[["vote_average", "vote_count"]] = pd.DataFrame(sc.fit_transform(tableau_movies[["vote_average", "vote_count"]] ))

#tableau_movies = tableau_movies.drop(tableau_movies_full[["vote_average", "vote_count"]], axis = 1)
#tableau_movies

##### Critère de Coude pour Kmeans movies
Inertie =[]
n_centroids = coude_centroid_movies
for i in range(1, n_centroids):
    kmeans = KMeans(n_clusters = i).fit(tableau_movies)
    #kmeans.fit(x)
    Inertie.append(kmeans.inertia_)
import matplotlib.pyplot as plt
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

fig = px.histogram(movies, x="Kmeans_movies_cluster", title = "Repartition des films par cluster Kmeans movies")
fig.update_xaxes(type='category')
fig.show()


link_user_votes = "data_csv/more/votecounts.csv"
# il existe des utilisateurs qui ont voté pour jusqu'à 18 000 films. On souhaite les tronquer
#trunc_user_high = 400
#trunc_user_low = 20
data_user_votes = pd.read_csv(link_user_votes)
data_user_votes = data_user_votes[ trunc_user_low  < data_user_votes['voteCount'] ]
data_user_votes = data_user_votes[  data_user_votes['voteCount'] < trunc_user_high]
df = data_user_votes.sort_values(by=['voteCount'])
fig = px.scatter(x=pd.Series(range(0,len(df['userId']))), y=df['voteCount'])
fig.show()


link_ratings = "data_csv/ratings.csv"
ratings = pd.read_csv(link_ratings)
ratings  =ratings.drop(['timestamp'], axis = 1)

ratings = ratings[np.isin(ratings['userId'], data_user_votes['userId'])] #on filtre les utilisateurs qui ont emis trop de votes, ou pas assez
ratings
#On observe le plus haut rating qu'un utlisateur a offert aux films qu'il a noté
maxrating = ratings.groupby(["userId"])["rating"].apply(lambda x : max(x)).reset_index()
#maxrating


fig = px.histogram(maxrating, x="rating", title = "Repartition du plus haut score offert par un utilisateur")
fig.update_xaxes(type='category')
fig.show()

#On observe combien de film ont noté les utilisateurs ayant offert un score maximal assez bas
df = data_user_votes[np.isin(data_user_votes['userId'], maxrating[maxrating['rating']<4]['userId'])]
df = df.sort_values(by = ['voteCount'])
fig = px.scatter(x=pd.Series(range(0,len(df['userId']))), y=df['voteCount'])
fig.show()



ratings_merge = pd.merge(ratings, movies, left_on ='movieId', right_on = 'id') #permet d'avoir une info sur le cluster de film en complément du tableau ratings
#ratings_merge

user_movies = ratings_merge.groupby(['userId', 'Kmeans_movies_cluster'])

df_score = user_movies['rating'].apply(lambda x : float(5) if float(5) in list(x) else 0).reset_index(name = 'score')
#df_score

df_users = df_score.pivot(index = 'userId', columns = 'Kmeans_movies_cluster').reset_index()
df_users = df_users.replace(np.nan, 0)
#tableau_users

df_kmeans_users = df_users['score']
df_kmeans_users


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
plt.title('Critere de Coude')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.show()


#### choix du nombre de clusters pour le Kmeans utilisateurs
kmeans = KMeans(n_clusters=kmeans_centroid_users).fit(df_kmeans_users)
centroids = kmeans.cluster_centers_

user_clusters = pd.DataFrame({'userId': tableau_users[('userId', '')], 'Kmeans_user_cluster': kmeans.labels_})
#user_clusters


fig = px.histogram(user_clusters, x="Kmeans_user_cluster", title = "Repartition des utilisateurs par cluster utilisateurs")
fig.update_xaxes(type='category')
fig.show()

#Maintenant, on souhaite retrouver la liste de films vues par un groupe d'utilisateurs
