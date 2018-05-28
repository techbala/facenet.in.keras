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

class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
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
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    for i in xrange(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images):  # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
                # all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]  # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    # print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)


def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1

    return image_paths, num_per_class


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

def embeddings(paths):
    triple_embeds = []
    X_input = Input((height, width, 3))
    X = InceptionResnetV2(X_input)
    model = Model(inputs=X_input, outputs=X)

    for triples_paths in paths:
        embeds = model.predict(np.stack(getImages(triples_paths)))
        triple_embeds.append(embeds)

    return triple_embeds


def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    anchor  = tf.Print( anchor, [anchor], "anchor", summarize=200)
    positive = tf.Print(positive, [positive], "positive", summarize=200)
    negative = tf.Print(negative, [negative], "negative", summarize=200)
    #pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=None)
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    pos_dist = tf.Print(pos_dist, [pos_dist], "pos_dist", summarize=200)
    #neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=None)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    neg_dist = tf.Print(neg_dist, [neg_dist], "neg_dist", summarize=200)
    basic_loss = tf.subtract(pos_dist, neg_dist) + alpha
    basic_loss = tf.Print(basic_loss, [basic_loss], "basic_loss", summarize=200)
    #loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    loss = tf.Print(loss, [loss], "loss", summarize=200)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_losses = tf.Print(regularization_losses, [regularization_losses], "regularization_losses", summarize=200)

    #total_loss = loss + regularization_losses
    #total_loss = tf.add_n([loss] + regularization_losses, name='total_loss')

    return loss

def saveTriplets(datasets):
    image_paths, num_per_class = sample_people(datasets, people_per_batch, images_per_person)
    nrof_examples = people_per_batch * images_per_person
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
    print( image_paths_array )
    emb_array = np.zeros((nrof_examples, embedding_size))
    embedList = embeddings(image_paths_array)

    lab = 0
    for each in embedList:
        for emb in each:
            emb_array[lab] = emb
            lab += 1

    triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class, image_paths, people_per_batch,0.2)

    triplet_paths = list(itertools.chain(*triplets))
    labels_array = np.reshape(np.arange(len(triplet_paths)), (-1, 3))
    triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths), 1), (-1, 3))
    print( triplet_paths_array.shape )
    print(triplet_paths_array.ndim )
    #np.savetxt( dataset_directory + "/triples.txt", triplet_paths_array, fmt=("%s","%s","%s"), delimiter="," )


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
    network_output = InceptionResnetV2(network_input)
    network = Model(inputs=network_input, outputs=network_output)
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


#datasets = get_dataset(dataset_directory + input_directory)
#saveTriplets(datasets)
#train()

triplets = np.loadtxt(dataset_directory + "/triples.txt", delimiter=",", dtype=np.str)
masterLabels = getMasterLabels(triplets)
train( triplets, masterLabels )























    ####################### Backup of old Code not sure about the order of execution. #######################
    # createFolder(filepath)
    # with open( filepath, 'w') as embedfile:
    # embedfile.write(model.predict(x ))
    # print( model.predict(x) )
    # print( type( model.predict(x)) )

    # img_path = '/ai/sources/ft/lfw.aligned/Tony_Blair/Tony_Blair_0009.png'
    # img_path ='/ai/sources/ft/lfw.aligned.small/Tom_Ridge/Tom_Ridge_0010.png'
    # img = image.load_img( img_path, target_size=(height,width))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)

    # dataset = get_dataset("/ai/sources/ft/lfw.aligned.small/Tiger_Woods")
    # for i in dataset:
    # print(i.__str__())

    # X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # X_train = X_train_orig/255.
    # X_test = X_test_orig/255.
    # Y_train = convert_to_one_hot(Y_train_orig, 6).T
    # Y_test = convert_to_one_hot(Y_test_orig, 6).T

    # input_shape = (height,width, 3)
    # X_input = Input( input_shape )

    # X = InceptionResnetV2(X_input)
    # print( X )

    # model = Model( inputs =X_input, outputs =X )
    # model.compile(loss='mse', optimizer='adam')

    # embed = Model( inputs = model.input, outputs = model.get_layer('embeddings').output)
    # print( model.predict(x))
    # model.summary()
    ####################### Backup of old Code not sure about the order of execution. #######################
