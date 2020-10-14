import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



#                                    Parametres
path = "/home/fitec/donnees_films/"
acp_dim = 26
k_means_movies = 8
k_means_users = 4






#chargement de la data,sélectionner uniquement les films qui ont été noté plus de 1000 fois, et les utilisateurs ayant noté moins de 1000 films

final_data_movie = pd.read_csv(path + "final_data_movie.csv")
ratings = pd.read_csv(path + "clean_ratings.csv")


all_movies = list(final_data_movie.drop_duplicates("id")["id"])
ratings = ratings[ratings["movieId"].isin(all_movies)]

nbr_votes_user = ratings.groupby("userId")["userId"].count().reset_index(name= "count_user")
ratings = pd.merge(ratings, nbr_votes_user, left_on="userId", right_on='userId', how='inner')
ratings = ratings[ratings["count_user"]<2000]

nbr_votes_movie = ratings.groupby("movieId")["movieId"].count().reset_index(name= "count_movie")
ratings = pd.merge(ratings, nbr_votes_movie, left_on="movieId", right_on='movieId', how='inner')
ratings = ratings[ratings["count_movie"]>1000]

legit_movies = list(ratings.drop_duplicates("movieId")["movieId"])
final_data_movie = final_data_movie[final_data_movie["id"].isin(legit_movies)]

del nbr_votes_user
del nbr_votes_movie
del legit_movies
del all_movies

#Supprimer vote average et vote count

data = final_data_movie.drop(["id", "title"], axis = 1)
#transformation – centrage-réduction
sc = StandardScaler()
data[["vote_average", "vote_count"]] = pd.DataFrame(sc.fit_transform(data[["vote_average", "vote_count"]] ))

# réfléchir à enlever vote count et vote average:
# - elle ne condorde pas avec un count sur ratings
# - cette information se retrouvera dans le cluster d'utilisateurs
# - cela ne devrait peut etre pas permettre de rapprocher les films entre eux : il y aura surement une cluster blockbusters et un cluster mauvais films  
data = data.drop(["vote_count", "vote_average"], axis = 1)










#0 ACP sur final_data_movie
#pca = PCA(n_components=110,svd_solver="arpack", random_state=80)
#pca_trans = pd.DataFrame(pca.fit_transform(data))
#
#def somme_ajoutees(liste):
#    retour = []
#    calcul = 0
#    for i in liste:
#        calcul += i
#        retour.append(calcul)
#    return retour
#
#explained_variances = somme_ajoutees(pca.explained_variance_ratio_)
#
#plt.scatter(x = [i-1 for i in range(1,len(explained_variances) + 1)],y = explained_variances)



#Il semblerait par la méthode du coude que les 25 premieres composantes sont à sélectionner pour résumer la donnée

pca = PCA(n_components=acp_dim, random_state = 80)
pca.fit(data)
data = pd.DataFrame(pca.transform(data))
del pca










#1 Realisation du kmeans sur final_data_movie
#méthode du coude pour déterminer le nombre de clusters
#inerties = []
#for i in range(1,15):
#    kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
#    inerties.append(kmeans.inertia_)
#
#plt.plot(inerties)

#on retient k = 8 selon le critere de coude

kmeans = KMeans(n_clusters=k_means_movies, random_state=0, n_init=10).fit(data)
final_data_movie["cluster_movie"] = kmeans.labels_
del kmeans
del data

#kmeans.n_iter_
# Les 10 meilleurs films de chaque cluster movie
#print(final_data_movie.groupby("cluster_movie")["cluster_movie"].count())
#movies = []
#clusters = []
#for i in range(k):
#    cluster = final_data_movie[final_data_movie["cluster_movie"] == i]
#    best_movies = cluster.sort_values("vote_average", ascending = False).head(30)
#    movies.append(list(best_movies["title"]))
#    clusters.append(i)
#d = {'clusters':clusters,'movies':movies}
#best_movies_cluster_movie = pd.DataFrame(d)

#description_clustering = final_data_movie.groupby(["cluster_movie"]).sum()



#   Rating clutser : ratings + clusters

#Merge de ratings et de final_data_movie (on garde seulement les clusters)
no_labels_final_data_movie = final_data_movie[["title", "id", "cluster_movie"]]
ratings = pd.merge(ratings, no_labels_final_data_movie, left_on='movieId', right_on='id')
ratings = ratings.drop("timestamp", axis = 1)
del no_labels_final_data_movie

# Pour faire le kmeans utilisateur, on retire les individus ayant noté moins de 50 films
#ratings = ratings_cluster[ratings["count_user"]>50]







# On veut obtenir la table pour lancer le kmeans utilisateurs

# moyenne et compte des notes par catégorie de film et par utilisateur
table = ratings.groupby(["userId", "cluster_movie"])["rating"].apply(lambda x : sum(x)/len(x)).reset_index(name = "mean")
table["compte"] = ratings.groupby(["userId", "cluster_movie"])["rating"].count().reset_index(name = "compte")["compte"]

#On fait un pivot pour obtenir la table des kmeans utilisateur et on renomme les colonnes(à la main)
final_data_user = table.pivot(index = "userId", columns ="cluster_movie", values=["mean","compte"]).reset_index()
final_data_user.columns = ["userId", "mean_0", "mean_1", "mean_2","mean_3","mean_4","mean_5","mean_6","mean_7", "compte_0", "compte_1", "compte_2", "compte_3","compte_4", "compte_5", "compte_6", "compte_7"]
#on remplace les nan par la moyenne de chaque catégorie de film, c'est environ 3.5 pour tous donc on remplace tous les nan par 3.5
#print(ratings.groupby("cluster_movie")["rating"].mean().reset_index(name="mean"))
final_data_user = final_data_user.fillna(3.5)

del table









#2 kmeans utilisateurs

#On calcule la moyenne des note spar utilisateur afin de ne pas se retrouver avec un clustering de type :
# groupe des utilisteurs qui notent bien, groupe des utilisateurs qui notent mal ,ect

moy_by_user = ratings.groupby("userId")["rating"].mean().reset_index(name="note_moy_user")
final_data_user = pd.merge(final_data_user, moy_by_user, left_on = "userId", right_on="userId")

data = final_data_user.drop(['userId', 'compte_0', 'compte_1','compte_2', 'compte_3','compte_4', 'compte_5','compte_6', 'compte_7' ], axis=1)

for col in data.drop("note_moy_user", axis=1).columns:
    data[col] = data[col]/data["note_moy_user"]
data = data.drop("note_moy_user", axis=1)

del moy_by_user

#méthode du coude pour déterminer le nombre de clusters
#inerties = []
#for i in range(1,30):
#    kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
#    inerties.append(kmeans.inertia_)
#    
#plt.plot(inerties)

kmeans = KMeans(n_clusters=k_means_users, random_state=0, n_init=10).fit(data)
final_data_user["cluster_user"] = kmeans.labels_

del data
del kmeans

# get user cluster on ratings
clusters = final_data_user.iloc[:,[0,-1]]
ratings = pd.merge(ratings, clusters, left_on="userId", right_on='userId', how='inner')






# lier cluster user et films
links = ratings.groupby(["cluster_user","movieId"])["rating"].count().reset_index(name="count")
moyenne = ratings.groupby(["cluster_user","movieId"])["rating"].mean().reset_index(name="mean")
links["mean"] = moyenne["mean"]



best_movies_per_cluster = links.sort_values(['cluster_user','mean'],ascending=False).groupby('cluster_user').head(100)
best_movies_per_cluster = pd.merge(best_movies_per_cluster, final_data_movie, left_on = "movieId", right_on = "id")[["title", "cluster_user","cluster_movie", "mean", "count"]].sort_values(["cluster_user", "mean"])



w = best_movies_per_cluster.groupby(["cluster_user", "cluster_movie"])["mean"].count().reset_index(name="count_per_cluster")

# A faire : recommander 3 films à chaque utilisateur à partir des films les plus aimés de son cluster et qu'il n'a pas vu





