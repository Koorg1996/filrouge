from finance import financial_data
from gettickerlist import get_tickers
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import math

tickers = get_tickers()


dataobjet = financial_data(tickers)

data = dataobjet.make_data()

data.to_csv('../data/datatodocker/finance.csv')