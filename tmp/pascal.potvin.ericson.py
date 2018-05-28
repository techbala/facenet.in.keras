from keras import backend as K
from keras.layers import Conv2D, Activation, Concatenate, Cropping2D, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense, Dropout
from keras.models import Sequential, Model
import numpy as np


def create_base_network(in_dims, out_dims):
    model = Sequential()
    model.add(BatchNormalization(input_shape=in_dims))
    model.add(Cropping2D(cropping=((11, 11), (11, 11))))
    model.add(Conv2D(32, 3, strides=2, name='conv1', padding='valid'))
    model.add(BatchNormalization(axis=3, name='bn1'))
    model.add(Activation('relu'))
    model.add(Dense(out_dims, activation='relu'))
    return model


in_dims = (321, 321, 3)
out_dims = 384

anchor_in = Input(shape=in_dims)
postive_in = Input(shape=in_dims)
negative_in = Input(shape=in_dims)

base_network = create_base_network(in_dims, out_dims)
anchor_out = base_network(anchor_in)
pos_out = base_network(postive_in)
neg_out = base_network(negative_in)
#merged_vector = np.concatenate((anchor_out, pos_out, neg_out))

model = Model(inputs=[anchor_in, postive_in, negative_in], outputs=[anchor_out, pos_out, neg_out ])

model.summary()
