import pandas as pd
import numpy as np
import os
#import json
#import ast
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


ratings = "data_csv/ratings.csv"
link = ratings
data = pd.read_csv(link)

#nombre total de films vues par utilisateur
votecount = data.groupby(["userId"])["rating"].apply(lambda x : len(list(x) )).reset_index(name = 'voteCount')
votecount


output_dir = "data_csv/more/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
votecounts.to_csv(str(output_dir+"votecounts.csv"))
#str(output_dir+"votecounts.csv")
