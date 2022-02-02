from typing import Tuple, List
import pandas as pd
import numpy as np
import itertools
import os
import json


def build_accesor_substring(dim, left_boundary, right_boundary):
    return f"{dim} >= {left_boundary} and {dim} < {right_boundary}"


def write_chunky(
    path,
    df: pd.DataFrame,
    image_shape: Tuple[int],
    image_chunksize: Tuple[int],
    chunk_columns: Tuple[str] = [],
    chunk_scale: Tuple[int] = None,
) -> None:
    """Writes a nested table to disk.

    A nested table follows the shape of the chunks of the paired image and
    contains the associated data points for each chunk of that image

    Args:
        path (str): The path to store to
        df (pd.DataFrame): The dataframe where the data points are stored
        image_shape (Tuple[int]): The shape of the image to follow
        image_chunksize (Tuple[int]): The shape of the chunks of the image to follow
        slice_dims (Tuple[str], optional): The columns that are use to infer chunks. Defaults to None.
        slice_scale (Tuple[int], optional): A scale that maps from pixelspace to columnspace Defaults to np.ones(len(slice_dims)).


    Raises:
        ValueError: [description]
    """

    if len(chunk_columns) > len(image_shape):
        raise ValueError(
            "The number of chunk_columns must be smaller the shape of the image"
        )

    chunk_scale = np.array(chunk_scale) if chunk_scale else np.ones(len(chunk_columns))

    follow_shape = np.array(image_shape)[: len(chunk_columns)]
    follow_chunksize = np.array(image_chunksize)[: len(chunk_columns)]
    follow_chunks = np.ceil(np.array((follow_shape / follow_chunksize)))

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, "meta.json"), "w") as f:
        f.write(
            json.dumps(
                {
                    "image_shape": image_shape,
                    "image_chunksize": image_chunksize,
                    "follow_chunks": follow_chunks.tolist(),
                    "follow_shape": follow_shape.tolist(),
                    "follow_chunksize": follow_chunksize.tolist(),
                    "chunk_scale": chunk_scale.tolist(),
                },
            )
        )

    for accessor in itertools.product(*[range(0, int(x)) for x in follow_chunks]):

        accessor_map = {
            chunk_columns[i]: (
                accessor[i] * follow_chunksize[i] * chunk_scale[i],
                (accessor[i] + 1) * follow_chunksize[i] * chunk_scale[i],
            )
            for i in range(len(accessor))
        }

        accesor_string = " and ".join(
            f"{dim} >= {left_boundary} and {dim} < {right_boundary}"
            for dim, (left_boundary, right_boundary) in accessor_map.items()
        )

        print(f"Writing Chunk of Dataset {path} with  {accesor_string}")
        puttable_df = df.query(accesor_string).sort_values(by=chunk_columns)

        csv_folder = os.path.join(path, *[str(i) for i in accessor[:-1]])
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder)
        puttable_df.to_csv(f"{csv_folder}/{accessor[-1]}.csv")
