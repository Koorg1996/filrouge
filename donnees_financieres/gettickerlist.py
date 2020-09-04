import pandas as pd

def get_tickers():
    tickerlistpath = '/home/fitec/Téléchargements/airlines.csv'
    df = pd.read_csv(tickerlistpath)
    tickers = df["Ticker"].tolist()
    return tickers



