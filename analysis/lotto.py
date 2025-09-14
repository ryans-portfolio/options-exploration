from dataclasses import dataclass
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#This is all historical so we can calculate PnL
@dataclass
class LottoTicket:
    def __init__(self, data: pl.DataFrame):
        data = data.sort("sip_timestamp")
        self.data = data
        self.symbol = data['symbol'][0]
        self.expiration = data['expiration'][0]
        self.strike = data['strike'][0]
        self.cp = data['cp'][0]
        self.num_trades = data.height
        self.expiration = data['expiration'][0]
        self.entry_price = data['price'][0]
        self.entry_dt = data['datetime'][0]
        self.data = data.with_columns(
            (pl.col('price') > self.entry_price).alias('is_up')

        )

    def plot_price_dist(self):
        fig = px.histogram(self.data, x="price", nbins=1000, color='is_up', title=f"Price Distribution for {self.symbol} {self.cp} {self.strike} Exp {self.expiration} - Entry {self.entry_price} at {self.entry_dt}")
        fig.show(renderer="browser")

    def return_stats(self):
        returns = self.data['return'].to_numpy()
        return {
            "mean": np.mean(returns),
            "std": np.std(returns),
            "min": np.min(returns),
            "max": np.max(returns),
            "median": np.median(returns),
            "25%": np.percentile(returns, 25),
            "75%": np.percentile(returns, 75),
        }