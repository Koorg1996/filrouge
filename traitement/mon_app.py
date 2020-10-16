import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import dash_html_components as html
import base64
from dash.dependencies import Input, Output

path = "/home/fitec/donnees_films/for_graphs/"

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

best_movies_per_cluster = pd.read_csv(path  + "best_movies_per_cluster.csv")
n_clusters = len(list(best_movies_per_cluster.groupby("Kmeans_user_cluster")["title"].count()))

cluster_genre = best_movies_per_cluster.groupby(["Kmeans_user_cluster", "total"])["total"].count().reset_index(name= "count")
    
cluster_genre = cluster_genre.groupby("Kmeans_user_cluster").head(20)

app.layout = html.Div(children=[
    html.H1(children='Resultats de la recommendation'),
    
    html.Div(children='''
        Choisir le cluster utilisateur à visualiser
    '''),
             
    dcc.RadioItems(
    id='cluster',
    options = [{'label': i, 'value': i} for i in range(n_clusters)],
    value = 0,
    labelStyle={'display': 'inline-block'}
    ),
    
    dcc.Slider(
    id='n_premiers',
    min=5,
    max=100,
    step=1,
    value=50
    ),
    
    dcc.Graph(id= "fig"),
    
    dcc.Graph(id= "fig2"),
    
    dcc.Graph(id = "fig3")
    

])
    
@app.callback(Output('fig', 'figure'),
              [Input('cluster', 'value'),
               Input('n_premiers', 'value')])

def make_chart(cluster,n_premiers): 
    df = best_movies_per_cluster[best_movies_per_cluster["Kmeans_user_cluster"]==cluster][0:n_premiers]
    count = df["count"].iloc[0]
    titre = "Part de vues des " + str(n_premiers) + " meilleurs films du cluster " + str(cluster) + " qui possède " + str(count) + " utilisateurs."
    if n_premiers > 50:
        fig = px.scatter(df, x= "title", y="part", title=titre)
    else:
        fig = px.bar(df, x= "title", y="part", title=titre)
    return fig

@app.callback(Output('fig2', 'figure'),
              [Input('cluster', 'value'),
               Input('n_premiers', 'value')])

def make_chart2(cluster,n_premiers): 
    df = best_movies_per_cluster[best_movies_per_cluster["Kmeans_user_cluster"]==cluster][0:n_premiers]
    count = df["count"].iloc[0]
    titre = "Moyenne des " + str(n_premiers) + " meilleurs films du cluster " + str(cluster) + " qui possède " + str(count) + " utilisateurs."
    if n_premiers > 50:
        fig = px.scatter(df, x= "title", y="mean", title=titre)
    else:
        fig = px.bar(df, x= "title", y="mean", title=titre)
    return fig

@app.callback(Output('fig3', 'figure'),
              [Input('cluster', 'value')])

def make_chart3(cluster):
    titre = "Distribution des genres pour les meilleurs films du cluster " + str(cluster)
    sur_ce_cluster = cluster_genre[cluster_genre["Kmeans_user_cluster"] == cluster]
    fig = px.pie(sur_ce_cluster, values = "count", names = "total", title= titre)
    return  fig

if __name__ == '__main__':
    app.run_server(debug=True)