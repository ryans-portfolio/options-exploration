import polars as pl
import numpy as np
import random

import pathlib
import os

this_dir = pathlib.Path(__file__).parent.resolve()
data_dir = os.path.join(this_dir, "..", "data")
trades_dir = os.path.join(data_dir, "trades")
output_dir = os.path.join(data_dir, "output")

