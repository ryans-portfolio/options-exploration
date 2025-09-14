import polars as pl
import numpy as np
import random

import pathlib
import os

from lotto import LottoTicket
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

this_dir = pathlib.Path(__file__).parent.resolve()
data_dir = this_dir / ".." / "data"
trades_dir = data_dir /  "2025"
output_dir = data_dir / "output"
root_dir = this_dir / ".."


def read_csvs(csv_files: list, random_sample: int | None = None, filter_func = None) -> pl.DataFrame: 
    if random_sample is not None:
        csv_files = random.sample(csv_files, random_sample)

    schema = {
        "ticker": pl.Utf8,
        "conditions": pl.Utf8,
        "correction": pl.Int64,
        "exchange": pl.Int64,
        "price": pl.Float64,
        "sip_timestamp": pl.Int64,
        "size": pl.Int64,
        }

    def helper(f, ff=None):
        print(f"Reading {f}")
        df = pl.read_csv(f, schema=schema)
        df = parse_tickers(df)
        if ff is not None:
            df = ff(df)
        return df

    return pl.concat([helper(f, ff=filter_func) for f in csv_files])

def parse_tickers(df: pl.DataFrame, ticker_col: str = "ticker", sip_col: str = "sip_timestamp") -> pl.DataFrame:
    # Regex to parse variable-length root
    pattern = r"^O:(?P<root>[A-Z]+)(?P<date>\d{6})(?P<cp>[CP])(?P<strike>\d+)$"

    # Parse ticker
    df = df.with_columns([
        pl.col(ticker_col).str.extract(pattern, 1).alias("symbol"),
        pl.col(ticker_col).str.extract(pattern, 2).alias("date"),
        pl.col(ticker_col).str.extract(pattern, 3).alias("cp"),
        pl.col(ticker_col).str.extract(pattern, 4).alias("strike"),
    ])

    # Convert date to Date type and strike to integer
    df = df.with_columns([
        pl.col("date").str.strptime(pl.Date, format="%y%m%d").alias("expiration"),
        pl.col("strike").cast(pl.Int64)
    ])

    # Efficiently convert from epoch nanoseconds to UTC datetime using numpy vectorized operations
    sip_np = df[sip_col].to_numpy()
    dt_np = sip_np.astype('datetime64[ns]')
    df = df.with_columns(pl.Series("datetime", dt_np))


    return df

def filter_df_by_col(filter_dict: dict) -> callable:
    def filter_func(df: pl.DataFrame) -> pl.DataFrame:
        for col, values in filter_dict.items():
            df = df.filter(pl.col(col).is_in(values))
        return df
    return filter_func

def entry_generator(data: pl.DataFrame):
    data = data.with_columns(
        pl.col("datetime").dt.round('30m').alias('RoundedTime')
    )

    entries = data.group_by(['ticker', 'RoundedTime']).agg([
        pl.col('price').median().alias('entry_price'),
        pl.col('datetime').first().alias('entry_dt'),
    ])
    
    entries = entries.select(['ticker','entry_price','entry_dt']).to_dicts()

    for i, entry in enumerate(entries):
        df = data.filter(
            (pl.col('ticker') == entry['ticker']) & 
            (pl.col('datetime') > entry['entry_dt'])
        ).sort('datetime')
        df = df.with_columns(
            (pl.col('price') / entry['entry_price'] - 1).alias('return'),
            pl.lit(entry['entry_dt']).alias('entry_dt'),
            pl.lit(entry['entry_price']).alias('entry_price'),
        )
        if df.is_empty():
            continue
        if (df['size'].sum() > 2000) & (df['entry_price'][0] < 0.75) & (df['entry_price'][0] > 0.04) & (df.height > 100):
            yield df
        # if i > 100000:
        #     break

def test_point(entry: pl.DataFrame):
    l = LottoTicket(entry)
    stats = l.return_stats()
    stats['entry_price'] = l.entry_price
    return stats

if __name__ == "__main__":
    import cProfile
    import multiprocessing as mp

    # pr = cProfile.Profile()
    # pr.enable()

    filter_dict = {
        "symbol": ["SPY"],
        "cp": ["C"],
    }

    filter_function = filter_df_by_col(filter_dict=filter_dict)

    last_10 = list(trades_dir.rglob('*.csv.gz'))
    last_10.sort(reverse=True)  # Sort files by name (newest first)
    last_10 = last_10[0:10]
    print(last_10)

    df = read_csvs(last_10, filter_func=filter_function)

    p = mp.Pool(mp.cpu_count())
    testpoints = p.map(test_point, entry_generator(df))
    
    data = pl.DataFrame(testpoints)
    print(data)
    data.write_parquet(output_dir / "recent_spy.parquet")

    fig = px.scatter_3d(data, x='entry_price', y='std', z='75%', color='median', hover_data=['entry_price', 'mean', 'std', 'min', 'max'])
    fig.show(renderer="browser")

    # pr.disable()
    # pr.dump_stats(os.path.join(root_dir, "profile_stats.prof"))

