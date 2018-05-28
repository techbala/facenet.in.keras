import numpy as np
import re

s = np.loadtxt( "triples.txt", delimiter=",", dtype=np.str)
print( len(s))
step = 64
batch_size = 64
partioned_path = s[(step * batch_size): ((step + 1) * batch_size), :]
print( partioned_path.shape )
print( s.shape )
print( s.ndim )

t = np.reshape(s,(-1))
print( type(t) )
for path in t:
    path = path.split("/")[-2]
    print( path )