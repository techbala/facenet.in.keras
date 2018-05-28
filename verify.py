from keras.models import model_from_json, load_model
from tensorflow import Graph, Session
from util import *
import random

def get_eval_embeddings( eval_datasets ):
    outer_graph = Graph()
    with outer_graph.as_default():
        outer_session = Session()
        with outer_session.as_default():
            """with open(parent_directory + model_directory + model_architecture_as_json, "r") as json_file:
                model_json = json_file.read()
            network = model_from_json( model_json )
            network.load_weights( parent_directory + model_directory + weights_of_model )
            """
            network = load_model( parent_directory + model_directory + complete_model, custom_objects={'triplet_loss':triplet_loss})
            embeds = network.predict(np.stack(getImages(eval_datasets)))
    return embeds

def notsame(datasets):
    data = []
    step = 0
    while step < 1000:
        first = random.choice(datasets)
        second = random.choice(datasets)
        if first != second:
            data += [ random.choice(first.image_paths), random.choice(second.image_paths)]
            step += 1
    return data

def same( datasets ):
    data = []
    step = 0
    while step < 1000:
        imageClass = random.choice(datasets)
        paths = imageClass.image_paths
        first = random.choice(paths)
        second = random.choice(paths)
        if first != second:
            data += [first, second]
            step += 1
    return data


def verify( data ):
    embeds = get_eval_embeddings( data )
    print( embeds.shape )
    dist = []
    for i in range( 0, len( embeds), 2):
        dist.append( np.linalg.norm(embeds[i] - embeds[i+1]) )

    print( dist )
    print( sum( i < 0.7 for i in dist ))
    print( np.histogram( dist, bins=np.arange( 0,1, 0.01)) )


def get_image_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        paths = dataset[i].image_paths
        folder_name = dataset[i].name
        for path in paths:
            image_paths_flat += [path]
            labels_flat += [ folder_name]
    return image_paths_flat, labels_flat

def split_dataset(dataset, min_nrof_images_per_class):
    train_set = []
    identity_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(ImageClass(cls.name, paths[:len(paths)-1]))
            identity_set.append(ImageClass(cls.name, paths[len(paths)-1:]))
    return train_set, identity_set

def who_is_it( train_encoding, train_label, identity_dict  ):
    min_dist = 100
    for name, encoding in identity_dict :
        dist = np.linalg.norm( encoding -  train_encoding)
        if train_label == name:
            print( " {} match: {} ".format( train_label, dist ))
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print( 'The given {} is NOT matched with the predicted {} and the distance is {} '.format( train_label, identity, str( min_dist )) )
    else:
        print('The given {} is matched with the predicted {} and the distance is {} '.format(train_label, identity,str(min_dist)))
        #print( "The expected distance " + str(np.linalg.norm( train_encoding - identity_dict[train_label] ) ))

    return min_dist, identity



datasets = get_dataset(parent_directory + evaluate_directory)
#s = same( datasets )
#ns = notsame(datasets)
#verify( s )
#verify( ns )

train_set, identity_set = split_dataset(datasets, 4)
train_paths, train_labels = get_image_labels(train_set)
identity_paths, identity_labels = get_image_labels(identity_set)
print( len( identity_labels) )
print( len( identity_paths) )
print( len(train_paths) )
print( len(train_labels) )
identity_embeds = get_eval_embeddings(identity_paths)
train_embeds = get_eval_embeddings( train_paths )
iden_dict  = [(identity,identity_embeds[idx] ) for idx, identity in enumerate(identity_labels)]
train_dict = [(identity,train_embeds[idx] ) for idx, identity in enumerate(train_labels)]
print( len(train_dict) )
print( len( iden_dict) )


for label, train_encoding in train_dict:
    who_is_it(train_encoding, label, iden_dict)