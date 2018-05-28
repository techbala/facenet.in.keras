import h5py
from h5py._hl.dataset import Dataset

from keras.models import load_model, model_from_json

"""with open('model.json', 'r') as f:
    network = model_from_json(f.read())
    network.load_weights( "weights_initial_facenet_model.h5" )


with h5py.File("weights_initial_facenet_model.h5", 'r' ) as hf:
    for i in hf.items():
        print(i[0], i[1])
        if type(i[1]) == 'tuple':
            for j in i[1].items():
                print(j[0], j[1])
                if type(j[1]) == 'tuple':
                    for k in j[1].items():
                        print(k[0], k[1])
                        if type(k[1]) == 'tuple':
                            for l in k[1].items():
                                print(l[0], l[1])
"""

filename = 'weights_initial_facenet_model.h5'
f = h5py.File(filename, 'r')

for key in f.keys():
    print( key )
    outer = f[key]
    for key in outer.keys():
        print( '    '+key )
        first = outer[key]
        for key in first.keys():
            print('         ' + key)
            second = first[key]
            if type(second) is not Dataset:
                for key in second.keys():
                    print('             ' + key)
                    third = second[key]
                    if type(third) is not Dataset:
                        for key in third.keys():
                            print('                 ' + key)
                            fourth = third[key]