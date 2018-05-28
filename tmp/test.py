
import os
import h5py
import errno
from numpy import genfromtxt
from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, Concatenate, Cropping2D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense, Dropout
from keras.preprocessing import image
from inceptionresnet import InceptionResnetV2
import itertools
import numpy as np

np.set_printoptions(threshold=np.inf)

input_shape = ( 321, 321, 3)
X_input = Input( input_shape )
X = InceptionResnetV2(X_input)
model = Model( inputs = X_input, outputs = X)

#model.save('model1.txt')
#with open('model.txt', 'w') as file:
#    file.write(model.to_json())

layer_name = 'embeddings'
inter_model = Model( inputs = model.input, outputs = model.get_layer(layer_name).output)

paths_3 = ['/ai/sources/keras.ins.facenet/lfw.more.than.nine/Abdullah_Gul/Abdullah_Gul_0001.jpg', '/ai/sources/keras.ins.facenet/lfw.more.than.nine/Abdullah_Gul/Abdullah_Gul_0002.jpg', '/ai/sources/keras.ins.facenet/lfw.more.than.nine/Abdullah_Gul/Abdullah_Gul_0003.jpg']

paths_2 = ['/ai/sources/keras.ins.facenet/lfw.more.than.nine/Abdullah_Gul/Abdullah_Gul_0001.jpg', '/ai/sources/keras.ins.facenet/lfw.more.than.nine/Abdullah_Gul/Abdullah_Gul_0002.jpg']

paths_1 = ['/ai/sources/keras.ins.facenet/lfw.more.than.nine/Abdullah_Gul/Abdullah_Gul_0001.jpg']


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std,1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
	
def createFolder(filePath):
    if not os.path.exists(os.path.dirname(filePath)):
        try:
                os.makedirs(os.path.dirname(filePath))
        except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                        raise

def getImages(paths):
	images = []
	for path in paths:
    		img = image.load_img(path, target_size=(321, 321))
    		x = image.img_to_array(img)
    		#x = np.expand_dims(x, axis=0)
    		#x = preprocess_input(x)
    		x = prewhiten(x)
    		#np.save('/ai/sources/keras.ins.facenet/image1.txt', x )
    		print( type(x))
    		print( x.shape)
    		print( x.ndim )
    		images.append( x )
	return images


#np.savetxt('/ai/sources/keras.ins.facenet/one1.txt', model.predict(np.stack(images)))

one = model.predict(np.stack(getImages(paths_1)))
two = model.predict(np.stack(getImages(paths_2)))
three = model.predict(np.stack(getImages(paths_3)))
#print(( one == two).all())
print(one)
print(two)
print(three)
