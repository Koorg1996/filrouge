import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


path = "/home/fitec/donnees_films/"

df = pd.read_csv(path + "final_data_movie.csv")




fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: Simulation de la modélisation et de la recommendation
    '''),

    dcc.Graph(
        id='Graph 1',
        figure=fig
        ,

    dcc.Graph(
        id='Graph 2',
        figure=fig
        ,

    dcc.Graph(
        id='Graph 3',
        figure=fig
        ,

    dcc.Graph(
        id='Graph 4',
        figure=fig
        
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)