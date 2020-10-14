import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import dash_html_components as html
import base64

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


image_filename = '/home/fitec/donnees_films/graphs/fig' 
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: Simulation de la modélisation et de la recommendation
    '''),

    html.Div(
    html.Img(src='data:image/png;base64,{}'.format(encoded_image)))

])

if __name__ == '__main__':
    app.run_server(debug=True)