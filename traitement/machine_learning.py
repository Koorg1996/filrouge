import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

path = "/home/fitec/donnees_films/"
final_data_movie = pd.read_csv(path + "final_data_movie.csv")
ratings = pd.read_csv(path + "clean_ratings.csv")


#1 Realisation du kmeans sur final_data_movie



data = final_data_movie.drop(["id", "title"], axis = 1)
#transformation – centrage-réduction
sc = StandardScaler()
data[["vote_average", "vote_count"]] = pd.DataFrame(sc.fit_transform(data[["vote_average", "vote_count"]] ))

# réfléchir à enlever vote count et vote average:
# - cette information se retrouvera dans le cluster d'utilisateurs
# - cela ne devrait peut etre pas permettre de rapprocher les films entre eux : il y aura surement une cluster blockbusters et un cluster mauvais films  
data = data.drop(["vote_count", "vote_average"], axis = 1)


l#méthode du coude pour déterminer le nombre de clusters
#inerties = []
#for i in range(1,20):
#    kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
#    inerties.append(kmeans.inertia_)
#
#plt.plot(inerties)

#on garde k=3, chaque film appartient alors à un des 3 clusters

k = 4
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(data)
final_data_movie["cluster_movie"] = kmeans.labels_
kmeans.n_iter_

# Les 10 meilleurs films de chaque cluster movie
final_data_movie.groupby("cluster_movie")["cluster_movie"].count()
movies = []
clusters = []
for i in range(k):
    cluster = final_data_movie[final_data_movie["cluster_movie"] == i]
    best_movies = cluster[cluster["vote_count"]>50].sort_values("vote_average", ascending = False).head(10)
    movies.append(list(best_movies["title"]))
    clusters.append(i)
d = {'clusters':clusters,'movies':movies}
best_movies_cluster_movie = pd.DataFrame(d)









#   Rating clutser : ratings + clusters

#Merge de ratings et de final_data_movie (on garde seulement les clusters)
join = final_data_movie[["title", "id", "cluster_movie"]]
ratings_cluster = pd.merge(ratings, join, left_on='movieId', right_on='id')
ratings_cluster = ratings_cluster.drop("timestamp", axis = 1)

# on sélectionne les utilisateurs ayant noté plus de 20 films et moins de 1000
nbr_votes = ratings_cluster.groupby("userId")["userId"].count().reset_index(name= "count")
enough_votes = list(nbr_votes[nbr_votes["count"] > 20]["userId"])
too_many_votes = list(nbr_votes[nbr_votes["count"] > 1000]["userId"])
ratings_cluster = ratings_cluster[ratings_cluster["userId"].isin(enough_votes)]
ratings_cluster = ratings_cluster[~ratings_cluster["userId"].isin(too_many_votes)]






# On veut obtenir la table pour lancer le kmeans utilisateurs

# moyenne et compte des notes par catégorie de film et par utilisateur
table = ratings_cluster.groupby(["userId", "cluster_movie"])["rating"].apply(lambda x : sum(x)/len(x)).reset_index(name = "mean")
table["compte"] = ratings_cluster.groupby(["userId", "cluster_movie"])["rating"].count().reset_index(name = "compte")["compte"]

#On fait un pivot pour obtenir la table des kmeans utilisateur et on renomme les colonnes(à la main)
final_data_user = table.pivot(index = "userId", columns ="cluster_movie", values=["mean","compte"]).reset_index()
final_data_user.columns = ["userId", "mean_0", "mean_1", "mean_2","mean_2", "compte_0", "compte_1", "compte_2", "compte_3"]
#on remplace les nan par la moyenne de chaque catégorie de film, c'est environ 3.5 pour tous donc on remplace tous les nan par 3.5
print(ratings_cluster.groupby("cluster_movie")["rating"].mean().reset_index(name="mean"))
final_data_user = final_data_user.fillna(3.5)









#2 kmeans utilisateurs



data = final_data_user.drop(['userId', 'compte_0', 'compte_1','compte_2', 'compte_3'], axis=1)

#méthode du coude pour déterminer le nombre de clusters
#inerties = []
#for i in range(1,10):
#    kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
#    inerties.append(kmeans.inertia_)
#plt.plot(inerties)


#on garde k=4, chaque user appartient alors à un des 4 clusters
k = 4
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(data)
final_data_user["cluster_user"] = kmeans.labels_


# Les 10 meilleurs films de chaque cluster
join = final_data_user.iloc[:,[0,-1]]
ratings_cluster = pd.merge(ratings_cluster, join, left_on="userId", right_on='userId', how='inner')






# A faire : Les 10 meilleurs films de chaque cluster user
ratings_cluster.drop_duplicates("userId").groupby("cluster_user")["cluster_user"].count()
movies = []
clusters = []
for i in range(4):
    cluster = ratings_cluster[ratings_cluster["cluster_user"] == i]
    movies = ratings_cluster.groupby("movieId")["rating"].mean().sort_values(ascending=False)
    titles = movies.append(list(best_movies["title"]))
    clusters.append(i)
d = {'clusters':clusters,'movies':movies}
best_movies_cluster_user = pd.DataFrame(d)


cluster = ratings_cluster[ratings_cluster["cluster_user"] == 0]
movies = ratings_cluster.groupby("movieId")["rating"].mean().sort_values(ascending=False)





