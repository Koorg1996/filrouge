import pandas as pd
import numpy as np
import json
import ast


path = "/home/fitec/donnees_films/"

# TRAVAIL NECESSAIRE SUR CHAQUE VARIABLE pour movies_metadata.csv

data = pd.read_csv(path + "movies_metadata.csv" )

# retirer les caracteres spéciaux
def delete_special_character(x):
    accepted_character = ['.',',','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'z', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'q', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'w', 'x', 'c', 'v', 'b', 'n', 'A', 'Z', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'Q', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'W', 'X', 'C', 'V', 'B', 'N', '{', '}', '[', ']', ':', ',', '_', '-', "'", '"']
    if type(x) == str:
        clean_str = ""
        for i, lettre in enumerate(x):
            if lettre not in accepted_character:
                clean_str += ""
            else:
                clean_str += lettre
        return clean_str
    else:
        return x

test = data.applymap(lambda x: delete_special_character(x))

#adult : ok
#belongs_to_collection
test["belongs_to_collection"] = test["belongs_to_collection"].apply(lambda x : "[]" if type(x) == float else x)
#budget
test["budget"] = test["budget"].apply(lambda x : float(x) if x not in ["ff9qCepilowshEtG2GYWwzt2bs4.jpg", "zV8bHuSL6WXoD6FWogP9j4x80bL.jpg", "zaSf5OG7V8X8gqFvly88zDdRm46.jpg"] else 0)
#genres : ok
#homepage
test["homepage"] = test["homepage"].apply(lambda x : "" if type(x) == float else x)
#id
test["id"] = test["id"].apply(lambda x : x if x not in ["1997-08-20", "2012-09-29", "2014-01-01"] else "")
test = test[test["id"] != ""]
test["id"] = test["id"].apply(lambda x : int(x))
#imdb_id
test["imdb_id"] = test["imdb_id"].apply(lambda x : "" if pd.isnull(x) else x)
#original_language
test["original_language"] = test["original_language"].apply(lambda x : "" if pd.isnull(x) else x)
#original_title : ok
#overview
test["overview"] = test["overview"].apply(lambda x : "" if pd.isnull(x) else x)
#popularity
test["popularity"] = test["popularity"].apply(lambda x : x if x not in ["BewareOfFrostBites"] else 0)
test["popularity"] = test["popularity"].apply(lambda x : 0 if pd.isnull(x) else float(x))
#poster_path
test["poster_path"] = test["poster_path"].apply(lambda x : "" if pd.isnull(x) else str(x))
#production_companies
test["production_companies"] = test["production_companies"].apply(lambda x : "[]" if pd.isnull(x) else x)
#production_countries
test["production_countries"] = test["production_countries"].apply(lambda x : "[]" if pd.isnull(x) else x)
#release_date
test["release_date"] = test["release_date"].apply(lambda x : "" if pd.isnull(x) else str(x))
#revenue
test["revenue"] = test["revenue"].apply(lambda x : 0 if pd.isnull(x) else float(x))
#runtime
test["runtime"] = test["runtime"].apply(lambda x : 0 if pd.isnull(x) else x)
### spoken_languages ok
test["spoken_languages"] = test["spoken_languages"].apply(lambda x : "" if pd.isnull(x) else str(x))
### status ok
test["status"] = test["status"].apply(lambda x : "" if pd.isnull(x) else str(x))
### tagline ok
test["tagline"] = test["tagline"].apply(lambda x : "" if pd.isnull(x) else str(x))
### title ok
test["title"] = test["title"].apply(lambda x : "" if pd.isnull(x) else str(x))
### video ok
test["video"] = test["video"].apply(lambda x : "" if pd.isnull(x) else str(x))
test["vote_average"] = test["vote_average"].apply(lambda x : 0 if pd.isnull(x) else float(x))
#vote_count
test["vote_count"] = test["vote_count"].apply(lambda x : 0 if pd.isnull(x) else int(x))


test.to_csv(path + "metadata_carac_speciaux.csv", index= False)

del data
del test
# TRAVAIL SUR keywords.csv

# retirer les caracteres spéciaux

keywords = pd.read_csv(path + "keywords.csv")
keywords = keywords.applymap(lambda x: delete_special_character(x))    
keywords.to_csv(path + "keywords_carac_speciaux.csv", index= False)

del keywords
# TRAVAIL sur ratings.csv

# enlever les doublons 
ratings = pd.read_csv(path + "ratings.csv")
ratings.dropna(subset=['userId'])
ratings=ratings.dropna(subset=['movieId'])
ratings=ratings.drop_duplicates()
ratings.to_csv(path + "clean_ratings.csv", index= False)

del ratings
