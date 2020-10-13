# Distribution des films selon le nombre de notes
def somme_ajoutees(liste):
    retour = []
    calcul = 0
    for i in liste:
        calcul += i
        retour.append(calcul)
    return retour

count_categories = []
num_film = len(final_data_movie)
for i in range(0,4000,50):
    j = i-50
    a = final_data_movie[final_data_movie["vote_count"] <= i]
    b = a[a["vote_count"] >= j]
    c = len(b)
    count_categories.append(c/num_film)
 
             
x = somme_ajoutees(count_categories)

plt.plot(count_categories)
