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

#'https://codepen.io/chriddyp/pen/bWLwgP.css'
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

best_movies_per_cluster = pd.read_csv(path  + "best_movies_per_cluster.csv")
best_movies_per_cluster["Kmeans_user_cluster"] = best_movies_per_cluster["Kmeans_user_cluster"].astype(str)
best_movies_per_cluster["Kmeans_movies_cluster"] = best_movies_per_cluster["Kmeans_movies_cluster"].astype(str)

n_clusters = len(list(best_movies_per_cluster.groupby("Kmeans_user_cluster")["title"].count()))

discrete_colors = ["blue", "darkslategray", "red", "orange", "cyan", "black", "lime", "coral", "green"]

nb_cluster = best_movies_per_cluster.groupby("Kmeans_user_cluster")[["nb_user_cluster", "Kmeans_user_cluster"]].head(1).reset_index(drop=True)


app.layout = html.Div(children=[
    html.H1(children='Resultats de la recommendation', 
            style={
            'textAlign': 'center',}
            ),
    
    html.Div(children='''
        Choisir le cluster utilisateur à visualiser
    '''),
             
    dcc.RadioItems(
    id='cluster',
    options = [{'label': i, 'value': str(i)} for i in range(n_clusters)],
    value = "0",
    labelStyle={'display': 'inline-block'}
    ),
    
    dcc.Slider(
    id='n_premiers',
    min=5,
    max=300,
    step=1,
    value=50
    ),
    
    html.H6(id="infos"),
    
    html.Div(children=[
            html.Div([
                    dcc.Graph(id= "fig")
                    ],
                    className="six columns"
                    ),
            html.Div([
                    dcc.Graph(id= "fig2")
                    ],
                    className="six columns"
                    )

    ], className="row"),
            
    html.Div(children=[
            html.Div([
                    dcc.Graph(id= "fig3")
                    ],
                    className="six columns"
                    ),
            html.Div([
                    dcc.Graph(id= "fig4")
                    ],
                    className="six columns"
                    )

    ], className="row"),
    
    html.Div([dcc.Graph(id="fig5")],
             style = {'display' : 'none'})
    

])
    
@app.callback(Output('fig', 'figure'),
              [Input('cluster', 'value'),
               Input('n_premiers', 'value')])

def make_chart(cluster,n_premiers): 
    df = best_movies_per_cluster[best_movies_per_cluster["Kmeans_user_cluster"]==cluster][0:n_premiers]
    titre = "Part de vues des meilleurs films."
    if n_premiers > 50:
        fig = px.scatter(df, x= "title", y="part", title=titre, color='Kmeans_movies_cluster',
                         color_discrete_map= {str(i) : discrete_colors[i] for i in range(n_clusters)})
    else:
        fig = px.bar(df, x= "title", y="part", title=titre, color='Kmeans_movies_cluster',
                     color_discrete_map= {str(i) : discrete_colors[i] for i in range(n_clusters)})
    return fig

@app.callback(Output('fig2', 'figure'),
              [Input('cluster', 'value'),
               Input('n_premiers', 'value')])

def make_chart2(cluster,n_premiers): 
    df = best_movies_per_cluster[best_movies_per_cluster["Kmeans_user_cluster"]==cluster][0:n_premiers]
    titre = "Variance des meilleurs films"
    if n_premiers > 50:
        fig = px.scatter(df, x= "title", y="variance", title=titre, color='Kmeans_movies_cluster',
                         color_discrete_map= {str(i) : discrete_colors[i] for i in range(n_clusters)})
    else:
        fig = px.bar(df, x= "title", y="variance", title=titre, color='Kmeans_movies_cluster',
                     color_discrete_map= {str(i) : discrete_colors[i] for i in range(n_clusters)})
    return fig

@app.callback(Output('fig3', 'figure'),
              [Input('cluster', 'value'),
               Input('n_premiers', 'value')])

def make_chart3(cluster, n_premiers):
    df = best_movies_per_cluster[best_movies_per_cluster["Kmeans_user_cluster"]==cluster][0:n_premiers]
    cluster_genre = df.groupby(["genre"])["genre"].count().reset_index(name= "count")

    titre = "Distribution des genres."
    fig = px.pie(cluster_genre, values = "count", names = "genre", title= titre)
    return  fig

@app.callback(Output('fig4', 'figure'),
              [Input('cluster', 'value'),
               Input('n_premiers', 'value')])

def make_chart4(cluster, n_premiers):
    df = best_movies_per_cluster[best_movies_per_cluster["Kmeans_user_cluster"]==cluster][0:n_premiers]
    cluster_count = df.groupby(["prod_count"])["prod_count"].count().reset_index(name= "count")
 
    titre = "Distribution des pays de production."
    fig = px.pie(cluster_count, values = "count", names = "prod_count", title= titre)
    return  fig

@app.callback(Output('fig5', 'figure'),
              [Input('cluster', 'value'),
               Input('n_premiers', 'value')])

def make_chart5(cluster, n_premiers):
    df = best_movies_per_cluster.groupby("Kmeans_user_cluster")[["title","mean","variance", "Kmeans_user_cluster"]].head(n_premiers)
    nb_dans_plusieurs_clusters = df.groupby("title")["title"].count().reset_index(name="nb").sort_values("nb", ascending=False)
    df = pd.merge(df, nb_dans_plusieurs_clusters, left_on = "title", right_on= "title")
    df = df[df["nb"]>n_clusters-1]
    
    fig = px.scatter(df, x= "mean", y="variance", color="Kmeans_user_cluster")
    return  fig


@app.callback(Output('infos', 'children'),
              [Input('cluster', 'value'),
               Input('n_premiers', 'value')])

def afficher_infos(cluster, n):
    nb = nb_cluster[nb_cluster["Kmeans_user_cluster"]==cluster]["nb_user_cluster"].iloc[0]
    return "Ci dessous les informations concernant les " + str(n) + " meilleurs films du groupe d'utilisateurs " + str(cluster) + ". Il est composé de " + str(nb) + " utilisateurs."

 
if __name__ == '__main__':
    app.run_server(debug=True)