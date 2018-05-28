import tensorflow as tf
import numpy as np
import os
import errno
from keras.layers import Input
from keras.models import Model
from keras.preprocessing import image
from inceptionresnet import InceptionResnetV2
import itertools
import re
from keras import backend as K
from keras.layers import Conv2D, Activation, Concatenate, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense, Dropout

height = 321
width = 321
people_per_batch = 90
images_per_person = 10
max_nrof_epochs = 2
epoch_size = 2
embedding_size = 128
batch_size = 2

dataset_directory = '/ai/sources/keras.ins.facenet'
input_directory = '/lfw.face.aligned.with.321'
embed_directory = '/lfw.embed.with.128'

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

def getImages(triples_paths):
    images = []
    for path in triples_paths:
        img = image.load_img(path, target_size=(height, width))
        x = image.img_to_array(img)
        x = prewhiten(x)
        images.append(x)
    return images


def getLabels(triples_paths, masterLabels):
    labels = []
    for path in triples_paths:
        path = path.split("/")[-1]
        path = re.sub("_\d*.png", "", path)
        labels.append(masterLabels.get(path))
    return labels

def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    anchor  = tf.Print( anchor, [anchor], "anchor", summarize=200)
    positive = tf.Print(positive, [positive], "positive", summarize=200)
    negative = tf.Print(negative, [negative], "negative", summarize=200)
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    pos_dist = tf.Print(pos_dist, [pos_dist], "pos_dist", summarize=200)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    neg_dist = tf.Print(neg_dist, [neg_dist], "neg_dist", summarize=200)
    basic_loss = tf.subtract(pos_dist, neg_dist) + alpha
    basic_loss = tf.Print(basic_loss, [basic_loss], "basic_loss", summarize=200)
    #loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    loss = tf.Print(loss, [loss], "loss", summarize=200)
    return loss

def getTriplets( triplets, masterLabels ):
    step = 0
    for line in triplets:
        images = np.stack(getImages(line))
        labels = getLabels( line, masterLabels )
        print( "Current processing step is %d" % ( step ))
        yield ( images, labels )
        step += 1

def train( triplets, masterLabels ):
    network_input = Input((height, width, 3))
    X = Cropping2D(cropping=((11, 11), (11, 11)))(network_input)
    X = Flatten()(X)
    X = Dense(128, name='embeddings')(X)
    network = Model(inputs=network_input, outputs=X)
    network.compile(loss=triplet_loss, optimizer='Adam', metrics=['accuracy'])
    print( network.losses)
    history = network.fit_generator(generator=getTriplets(triplets, masterLabels), epochs=max_nrof_epochs, steps_per_epoch=4050,
                                    verbose=2, max_queue_size=1)
    history_dict = history.history
    for key, value in history_dict.items():
        print('Key is %s and value is %s' % (key, value))
    print(history_dict.keys())

def getMasterLabels( triplets ):
    dict ={}
    token = 1
    for line in triplets:
        for path in line:
            path = path.split("/")[-1]
            path = re.sub("_\d*.png", "", path)
            if path not in dict :
                dict[ path ] = token
                token += 1
    return dict


triplets = np.loadtxt(dataset_directory + "/triples.txt", delimiter=",", dtype=np.str)
masterLabels = getMasterLabels(triplets)
train( triplets, masterLabels )