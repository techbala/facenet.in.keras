import numpy as np
from keras.preprocessing import image

triplets = np.loadtxt("triples.check.txt", delimiter=",", dtype=np.str)

def getImages(triples_paths):
    images = []
    for path in triples_paths:
        x = np.zeros((321,321))
        images.append(x)
    return images

for i in range(10):
    for line in triplets:
        images = np.stack(getImages(line))
        print( images.ndim )
        print( images.shape )