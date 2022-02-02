from chunky.read import read_chunky

x = read_chunky("data/chunkied")
t = x[:200, :200, :, :, :]
print(t)
