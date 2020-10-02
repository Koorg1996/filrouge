
#Realisation du kmeans sur final_data_movie


## import de ratings.csv et merge avec le résultat du kmeans

ratings.dropna(subset=['userId'])
ratings=ratings.dropna(subset=['movieId'])
ratings=ratings.drop_duplicates()

final_dataset = ratings.merge(final_data_movie, left_on='movieId', right_on='id', how='inner')


#df_movie[[ 'id', 'movieId', 'title']].head(5)   !!!!!
####FIN