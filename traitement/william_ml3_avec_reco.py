#!/usr/bin/python3

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import datetime
from matplotlib.backends.backend_pdf import PdfPages
################### Parametres simulation ##############################
remove_col_kmeans_movies = ['id','title','vote_average', 'vote_count']
trunc_user_high = 11500 #nombre max de vues total par user
trunc_user_low = 20 #nombre min de vues total par user
trunc_movie_low = 7
trunc_movie_high = 76000
coude_centroid_movies = 14
kmeans_centroid_movies = 4
coude_centroid_users  =9
kmeans_centroid_users = 4
n=5 #nbr de films à recommander
################### Parametres de Modelisation ##############################
p_c_a = False # Activer ou pas la Principal Component analysis
acp_dim = 26 # Principal Component analysis

###### Modelisation Kmeans utlisateur #########
a=0.8
# valeur de a doit être comprise entre zero et un!!!
# Pour a = 0,
# Si l'utilisateur a offert un rating de 5 a un des films. score vaut 5
# Si l'utilisateur n'a offert un rating de 5 a aucun un des films. score vaut le score moyen qu'il a offert aux films
# Pour a =1,
# Le score vaut toujours le score moyen qu'il a offert aux films

#plus a augmente et se rapproche de 1, plus l'ecart entre avoir son film préfére dans un cluster ou pas diminue

##########################################################################
date_time = datetime.datetime.now()
################### fichier output ##############################
output_dir = "data_csv/more/output_"+str(date_time) +'/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
content_file =  'Colonnes retirées de tableau_movies'  + str(remove_col_kmeans_movies) +'\n'
content_file += 'On retire les utilisateurs ayant vu plus de ' + str(trunc_user_high) +' films' +'\n'
content_file += 'On retire les utilisateurs ayant vu moins de ' + str(trunc_user_low) +' films' +'\n'
content_file += 'On retire les films vus plus de ' + str(trunc_movie_high) +' fois' +'\n'
content_file += 'On retire les films vus moins de ' + str(trunc_movie_low) +' fois' +'\n'
content_file += 'Kmeans utilisateur avec parametre de combinaison linéaire ' + str(a) +' ' +'\n'
content_file += ' Nombre de centroides Kmeans Movies ' + str(kmeans_centroid_movies) +' ' +'\n'
content_file += ' Nombre de centroides Kmeans utilisateur ' + str(kmeans_centroid_users) +' ' +'\n'
if p_c_a:
    content_file += 'Principal Component Analysis activée et réduit à  ' + str(acp_dim ) + ' dimensions' + '\n'
else:
    content_file += 'Principal Component Analysis inactive  '

f = open(output_dir+"output"+str(date_time)+".txt", "a")
f.write(content_file)
f.close()

################### fichier pdf ##############################
pp = PdfPages(output_dir+'Récapitulatif graphiques '+str(date_time)+'.pdf')
firstPage = plt.figure(figsize=(11.69,8.27))
firstPage.clf()
txt = content_file
firstPage.text(0.5,0.5,txt, transform=firstPage.transFigure, size=12, ha="center")
pp.savefig()





################### fichier input ###############################
input_dir = "data_csv/"
#input_dir = "/home/fitec/donnees_films/"
################################################################












################ Lecture et tri de la donnée ########################
tableau_movies_full = pd.read_csv(input_dir + "final_data_movie.csv", index_col=0)
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
ratings["title"] = pd.merge(ratings, tableau_movies_full, left_on = "movieId", right_on="id")["title"]
######################################################################








######################## FIGURE 1 ##############################
fig1 = px.scatter(x=pd.Series(range(0,len(df['userId']))), y=df['voteCount'], title = "Nombre de vues total par utilisateur selectionné")
fig1.write_html(output_dir + "Nombre de vues total par utilisateur.html")







######################## Differents graphs pour illustrer ##############################
#On observe le plus haut rating qu'un utlisateur a offert aux films qu'il a noté
maxrating = ratings.groupby(["userId"])["rating"].apply(lambda x : max(x)).reset_index()

fig3 = px.histogram(maxrating, x="rating", title = "Repartition du plus haut score offert par un utilisateur")
fig3.update_xaxes(type='category')

#On observe combien de film ont noté les utilisateurs ayant offert un score maximal assez bas
#df = data_user_votes[np.isin(data_user_votes['userId'], maxrating[maxrating['rating']<4]['userId'])]
#df = df.sort_values(by = ['voteCount'])
#fig4 = px.scatter(x=pd.Series(range(0,len(df['userId']))), y=df['voteCount'])




del data_user_votes
#####################################################################################


####################### Principle Component Analysis #####################################

if p_c_a:
    pca = PCA(n_components=acp_dim, random_state=80)
    pca.fit(tableau_movies)
    tableau_movies = pd.DataFrame(pca.transform(tableau_movies))

####################### Kmeans sur le récapiptulatif de films #############################

##### Critère de Coude pour Kmeans movies


######################## COUDE KMEANS MOVIES ##############################
Inertie =[]
n_centroids = coude_centroid_movies
for i in range(1, n_centroids):
    kmeans = KMeans(n_clusters = i).fit(tableau_movies)
    Inertie.append(kmeans.inertia_)

coude_movies = plt.figure()
plt.plot(range(1, n_centroids), Inertie)
plt.title('Critere de Coude Kmeans movies')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
coude_movies.show()
pp.savefig(coude_movies)

#### on lance kmeans avec k clusters défini en parametre de simulation
kmeans = KMeans(n_clusters=kmeans_centroid_movies).fit(tableau_movies)
centroids = kmeans.cluster_centers_

movies = pd.DataFrame({'id': tableau_movies_full['id'], 'Kmeans_movies_cluster': kmeans.labels_})

# On ajoute le clustering à la table ratings
ratings = pd.merge(ratings, movies, left_on = "movieId", right_on = "id")
tableau_movies_full = pd.merge(tableau_movies_full, movies, left_on = "id", right_on = "id")

###### Graph de la répartition par cluster de film ######
####### Plotly
fig2 = px.histogram(movies, x="Kmeans_movies_cluster", title = "Repartition des films par cluster Kmeans movies ")
fig2.update_xaxes(type='category')

fig2.write_html(output_dir + "Repartition des films par cluster Kmeans movies.html")
####### matplotlib pour le pdf
movies.groupby('Kmeans_movies_cluster').count().plot(kind='bar', title = "Repartition des films par cluster Kmeans movies ")
plt.ylabel('Nombre de films')
pp.savefig()
#########################################################################################







######################## FIGURE 3 ##############################
####### Plotly
# fig3 = px.histogram(maxrating, x="rating", title = "Repartition du plus haut score offert par un utilisateur")
# fig3.update_xaxes(type='category')
# fig3.write_html(output_dir + "Repartition du plus haut score offert par un utilisateur.html")
####### matplotlib pour le pdf
maxrating.groupby('rating').count().plot(kind='bar', title = "Repartition du plus haut score offert par un utilisateur")
plt.ylabel("Nombre d'utilisateurs ")
pp.savefig()

del maxrating





######################## FIGURE  4 ##############################
fig4 = px.scatter(x=pd.Series(range(0,len(df['userId']))), y=df['voteCount'],title = "Nombre de films vues par utilisateur ayant offert un score maximal bas")
fig4.write_html(output_dir + "Nombre de films vues par utilisateur ayant offert un score maximal bas.html")





################ Traitement pour le kmeans users ##############################################

#permet d'avoir une info sur le cluster de film en complément du tableau ratings
user_movies = ratings.groupby(['userId', 'Kmeans_movies_cluster'])
df_score = user_movies['rating'].apply(lambda x : (1-a)*float(5) + a*sum(list(x))/len(list(x)) if float(5) in list(x) else a*sum(list(x))/len(list(x))).reset_index(name = 'score')
df_users = df_score.pivot(index = 'userId', columns = 'Kmeans_movies_cluster').reset_index()
df_users = df_users.replace(np.nan, 0)

df_kmeans_users = df_users['score']

del df_score

#################################################################################################








############################## Kmeans utilisateurs #######################################


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
pp.savefig(coude_users)


#### choix du nombre de clusters pour le Kmeans utilisateurs
kmeans = KMeans(n_clusters=kmeans_centroid_users).fit(df_kmeans_users)
centroids = kmeans.cluster_centers_

user_clusters = pd.DataFrame({'userId': df_users[('userId', '')], 'Kmeans_user_cluster': kmeans.labels_})
# On ajoute le clustering à la tale ratings
ratings = pd.merge(ratings, user_clusters, left_on="userId", right_on="userId")

###### Graph de la répartition par cluster des users ######
fig5 = px.histogram(user_clusters, x="Kmeans_user_cluster", title = "Repartition des utilisateurs par cluster utilisateurs")
fig5.update_xaxes(type='category')

fig5.write_html(output_dir + "Repartition des utilisateurs par cluster utilisateurs.html")

user_clusters.groupby('Kmeans_user_cluster').count().plot(kind='bar', title = "Repartition des utilisateurs par cluster utilisateurs")
plt.ylabel("Nombre d'utilisateurs ")
pp.savefig()

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
links["mean"] = moyenne["mean"]

### récupérer les meilleurs films par cluster user selon la mean
parmi_combien = 100
best_movies_per_cluster = links.sort_values(["Kmeans_user_cluster",'mean'],ascending=False).groupby("Kmeans_user_cluster").head(parmi_combien).reset_index(drop=True)
best_movies_per_cluster["nb_user_cluster"] = pd.merge(best_movies_per_cluster, nb_users_cluster, left_on="Kmeans_user_cluster", right_on = "Kmeans_user_cluster")["rating"]
best_movies_per_cluster["part"] = (best_movies_per_cluster["count"] / best_movies_per_cluster["nb_user_cluster"]) * 100
best_movies_per_cluster = pd.merge(best_movies_per_cluster, tableau_movies_full, left_on = "movieId", right_on = "id")[["title", "Kmeans_user_cluster","Kmeans_movies_cluster", "mean", "part"]].sort_values(["Kmeans_user_cluster", "mean"], ascending = False).reset_index()

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





############################" Recommendations pour chaque utilisateur ###############################
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


##################  FIN DE SIMULATION ###########
pp.close()
date_timeend = datetime.datetime.now()
runtime = date_timeend - date_time