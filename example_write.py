from chunky.write import write_chunky
import pandas as pd

df = pd.read_csv("data/loc-smlm.csv", delimiter="\t")
newdf = df.rename(
    columns={"x[nm]": "x", "y[nm]": "y", "z[nm]": "z", "intensity[a.u.]": "intensity"}
)

write_chunky(
    "data/chunkied",
    newdf,
    (2048, 2048, 20, 1, 1),
    (500, 500, 20, 1, 1),
    chunk_columns=["x", "y"],
    chunk_scale=[16, 16],
)
