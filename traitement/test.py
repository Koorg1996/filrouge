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






#on filtre les utilisateurs qui ont emis trop de votes, ou pas assez dans ratings
data_user_votes = ratings.groupby(["userId"])["rating"].apply(lambda x : len(list(x) )).reset_index(name = 'voteCount')
data_user_votes = data_user_votes[ trunc_user_low  < data_user_votes['voteCount'] ]
data_user_votes = data_user_votes[  data_user_votes['voteCount'] < trunc_user_high]
df = data_user_votes.sort_values(by=['voteCount'])


######################## FIGURE 1 ##############################
fig1 = px.scatter(x=pd.Series(range(0,len(df['userId']))), y=df['voteCount'])


