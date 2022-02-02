#%%
import dask.array as da
import pandas as pd
import dask.dataframe as dd
import numpy as np
import os
import math
import itertools
import json

#%%


class ChunkyReader:
    def __init__(self, table_path) -> None:

        with open(os.path.join(table_path, "meta.json")) as f:
            self.table_meta = json.loads(f.read())
            self.image_shape = self.table_meta["image_shape"]
            self.chunk_size = self.table_meta["image_chunksize"]
            self.follow_shape = self.table_meta["follow_shape"]

        self.table_path = table_path

    def __getitem__(self, subscript):
        assert len(subscript) == len(
            self.image_shape
        ), "Slicing a shape different to the image that was used to create the table"

        df_range = []
        df = None

        for chunk_size, slice_, shape in zip(
            self.chunk_size, subscript, self.follow_shape
        ):
            assert isinstance(slice_, slice), "Only slices are allowed"

            start = slice_.start if slice_.start is not None else 0
            stop = slice_.stop if slice_.stop is not None else shape

            first = math.floor(start / chunk_size)
            last = math.ceil(stop / chunk_size)
            df_range.append(range(first, last))

        for accessor in itertools.product(*df_range):

            chunk_folder = os.path.join(
                self.table_path, *[str(i) for i in accessor[:-1]]
            )
            file = f"{chunk_folder}/{accessor[-1]}.csv"
            newdf = pd.read_csv(file)
            if df is None:
                df = newdf
            else:
                df = pd.concat([newdf, df])

        return df


def read_chunky(filepath: str):
    """Reads chunks

    Args:
        filepath (str): [description]
    """
    return ChunkyReader(filepath)
