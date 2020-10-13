import pandas as pd
import numpy as np
import json
import ast


fichier = "keywords.csv"
link = "/home/fitec/donnees_films/"+fichier
data = pd.read_csv(link)

df = data

def add(x, total):
    items = []
    if type(x) == str and x[0] == "[":
        a = ast.literal_eval(x)
        if len(a) > 0:
            for j in range(len(a)):
                comp = a[j]["name"]
                items.append(comp)
        else:
            items.append("null")
    else:
        items.append("null")
    total.append(items)

def get_one_of_k(liste, first_elements):
    for item in first_elements:
        if item in liste:
            return 1   
    return 0
        
def analyse_dictio(col, max_k = 100):
    ID = df['id']
    col = df[col]
    total_obj = []
    
            
    col.apply(add, total = total_obj)
    all_in_list = pd.DataFrame([i for l in total_obj for i in l], columns = ["objets"])
    count = all_in_list.groupby("objets")["objets"].count().reset_index(name="count").sort_values("count", ascending = False)
    
    k = len(count["count"])
    if k > max_k:
        k = max_k
    
    for i in range(2,k,2):
        first_k = count["objets"].tolist()[:i]
        
        result = []
        n = 0
        
        for li in total_obj:
            x = get_one_of_k(li, first_elements=first_k)
            if x == 1:
                n += 1
            result.append(x) 
            
        print("Si on prend les "+str(i)+" premiers genres "+str(n/len(total_obj)*100)+" % des films comportent au moins un de ces genres ") 
        print(first_k) 

analyse_dictio("keywords")
        

