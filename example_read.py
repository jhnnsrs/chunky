from chunky.read import read_chunky

x = read_chunky("data/chunkied")

# Indexing is handled as if indexing in the image
t = x[:200, :200, :, :, :]
print(t)
