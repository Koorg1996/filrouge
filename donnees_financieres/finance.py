import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import math

class financial_data():
    def __init__(self, entreprises):
        self.entreprises = entreprises
        

    def make_plots(self, m):
        #METADATA
        # m = final number of columns
        n = math.ceil(len(self.entreprises)/m)
        #l and c : count rows and columns to display to right subplot
        l = 1
        c = 1
        fig = make_subplots(rows=n, cols=m)

        for i, entreprise in enumerate(self.entreprises):
            ent = yf.Ticker(entreprise)
            data = ent.history(period="max",start="2019-01-12")
            data = data.reset_index()[["Date", "Open"]]
            if m == c:
                fig.add_trace(
                    go.Scatter(x = data["Date"], y= data["Open"]),
                    row = l, col=c
                )
                l += 1
                c = 1
            elif m != c:
                fig.add_trace(
                    go.Scatter(x = data["Date"], y= data["Open"]),
                    row = l, col=c
                )
                c += 1

        return fig
    
    def make_data(self):
        df = pd.DataFrame(columns=['Date'])

        for i, entreprise in enumerate(self.entreprises):
            ent = yf.Ticker(entreprise)
            data = ent.history(period="max",start="2019-01-12")
            data = data.reset_index()[["Date", "Open"]]
            if i == 0:
                df["Date"] = data["Date"]
            df = df.merge(data, left_on="Date", right_on="Date")
        names = ["Date"] + self.entreprises
        df.columns = names

        return df





