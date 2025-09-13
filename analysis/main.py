import polars as pl
import numpy as np
import random

import pathlib
import os

this_dir = pathlib.Path(__file__).parent.resolve()
data_dir = os.path.join(this_dir, "..", "data")
trades_dir = os.path.join(data_dir, "2025")
output_dir = os.path.join(data_dir, "output")
root_dir = os.path.join(this_dir, "..")


def read_csvs(dir_path: str, random_sample: int | None = None, filter_func = None) -> pl.DataFrame:
    csv_files = list(pathlib.Path(dir_path).rglob("*.csv.gz"))     
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


if __name__ == "__main__":
    import cProfile

    pr = cProfile.Profile()
    pr.enable()

    filter_dict = {
        "symbol": ["SPY", "SPX"],
        "cp": ["C"],
    }

    filter_function = filter_df_by_col(filter_dict=filter_dict)

    df = read_csvs(trades_dir, random_sample=10, filter_func=filter_function)

    print(df)

    pr.disable()
    pr.dump_stats(os.path.join(root_dir, "profile_stats.prof"))

