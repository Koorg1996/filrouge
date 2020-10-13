import pandas as pd
import numpy as np
import os
#import json
#import ast
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px

ratings = "data_csv/ratings.csv"
link = ratings
data = pd.read_csv(link)

#nombre total de films vues par utilisateur
file_user_votecounts = data.groupby(["userId"])["rating"].apply(lambda x : len(list(x) )).reset_index(name = 'voteCount')
file_user_votecounts 


fig = px.scatter(x = pd.Series(range(0,len(file_user_votecounts['userId']))), y = file_user_votecounts['voteCount'].sort_values()
                 , labels=dict( x="users", y="User_votecount") ,title = "Repartition du nombre total de vues par utilisateur" )


fig.show()

output_dir = "data_csv/more/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
file_user_votecounts.to_csv(str(output_dir+"votecounts.csv"))
#str(output_dir+"votecounts.csv")

#nombre total vues par film
file_movie_votecounts = data.groupby(["movieId"])["rating"].apply(lambda x : len(list(x) )).reset_index(name = 'vote_count')
file_movie_votecounts


output_dir = "data_csv/more/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
file_movie_votecounts.to_csv(str(output_dir+"votecounts_movies.csv"))
str(output_dir+"votecounts.csv")

#nombre total de vues par film
fig = px.scatter(x = pd.Series(range(0,len(file_movie_votecounts['movieId']))), y = file_movie_votecounts['vote_count'].sort_values()
                 , labels=dict( x="movie", y="Movie_votecount") ,title = "Repartition du nombre total de vues par film" )


fig.show()
