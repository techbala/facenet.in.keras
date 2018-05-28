from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import Adam
from tensorflow import Graph, Session
from inceptionresnet import InceptionResnetV2
import itertools
from util import *


def buildTripletPairs( datasets, filename ):
    embed_graph = Graph()
    triplet_paths_array = []
    with embed_graph.as_default():
        X_input = Input((height, width, 3))
        X = InceptionResnetV2(X_input)
        model = Model(inputs=X_input, outputs=X)
        with tf.Session(graph=embed_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for file_inc in range( max_nrof_epochs ):
                image_paths, num_per_class = sample_people(datasets, people_per_batch, images_per_person)
                nrof_examples = people_per_batch * images_per_person
                emb_array = np.zeros((nrof_examples, embedding_size))
                embeds = model.predict(np.stack(getImages(image_paths)))
                for loc in range(nrof_examples):
                    emb_array[loc, :] = embeds[loc]
                triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class, image_paths, people_per_batch,0.2)
                triplet_paths = list(itertools.chain(*triplets))
                triplet_paths_array.extend(np.reshape(np.expand_dims(np.array(triplet_paths), 1), (-1, 3)))


    np.savetxt( filename, triplet_paths_array, fmt=("%s", "%s", "%s"), delimiter=",")

def getTriplets(saved_triplet_paths, masterLabels, ):
    steps = 0
    while True:
        """
        image_paths, num_per_class = sample_people(datasets, people_per_batch, images_per_person)
        nrof_examples = people_per_batch * images_per_person
        emb_array = np.zeros((nrof_examples, embedding_size))
        embedList = embeddings(image_paths)
        for i in range(nrof_examples):
            emb_array[i, :] = embedList[i]

        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class, image_paths, people_per_batch,
                                                                    0.2)
        triplet_paths = list(itertools.chain(*triplets))
        """
        start = steps*training_images_per_step
        end = (steps+1) * training_images_per_step
        print( "start:%d and end:%d" %( start, end ))
        step_records = saved_triplet_paths[start:end]
        triplet_paths = np.reshape(step_records, (-1))
        triplet_image_pixels = np.stack(getImages(triplet_paths))
        triplet_image_labels = np.stack(getLabels(triplet_paths, masterLabels))
        yield (triplet_image_pixels, triplet_image_labels)
        steps += 1
        print("======================> current step is " + str( steps ))


class MyDebugWeights(Callback):
    def __init__(self):
        super(MyDebugWeights, self).__init__()
        self.weights = []
        self.tf_session = K.get_session()

    """
    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            name = layer.name
            for i, w in enumerate(layer.weights):
                w_value = w.eval(session=self.tf_session)
                w_norm, w_mean, w_std = calc_stats(np.reshape(w_value, -1))
                self.weights.append((epoch, "{:s}/W_{:d}".format(name, i),
                                     w_norm, w_mean, w_std))
    """

    def on_train_end(self, logs=None):
        for e, k, n, m, s in self.weights:
            print("{:3d} {:20s} {:7.3f} {:7.3f} {:7.3f}".format(e, k, n, m, s))

debug = MyDebugWeights()

def train(masterLabels, filename):
    outer_graph = Graph()
    with outer_graph.as_default():
        outer_session = Session()
        with outer_session.as_default():
            network_input = Input((height, width, 3))
            network_output = InceptionResnetV2(network_input)
            network = Model(inputs=network_input, outputs=network_output)
            print( network.summary( ))
            opt = Adam( lr=0.01, beta_1=0.99, beta_2=0.999, epsilon=0.1)
            network.compile(loss=triplet_loss, optimizer=opt, metrics=['accuracy'])
            saved_triplet_paths = np.loadtxt(filename, delimiter=",", dtype=np.str)
            steps_for_each_epoch = len( saved_triplet_paths) // (max_nrof_epochs * training_images_per_step)
            print( ' steps per each epoch is %d ' %( steps_for_each_epoch ))
            checkpoint = ModelCheckpoint( parent_directory+model_directory+interim_best_weights_of_model, monitor='loss', verbose=1, save_best_only=True, mode='max' )
            history = network.fit_generator(generator=getTriplets(saved_triplet_paths, masterLabels), epochs=max_nrof_epochs,
                                            steps_per_epoch= steps_for_each_epoch,
                                            verbose=2, max_queue_size=1, callbacks=[checkpoint, debug])
            history_dict = history.history
            for key, value in history_dict.items():
                print('Key is %s and value is %s' % (key, value))
            print(history_dict.keys())
            model_json = network.to_json()
            with open( parent_directory + model_directory + model_architecture_as_json, "w") as json_file:
                json_file.write(model_json)
            network.save(parent_directory + model_directory + complete_model)
            network.save_weights( parent_directory + model_directory + weights_of_model)

def execute_train():
    datasets = get_dataset(parent_directory + input_directory)
    masterLabels = getMasterLabels(datasets)
    filename = parent_directory + triples_directory + "/triples.txt"
    createFolder( filename )
    buildTripletPairs( datasets, filename )
    train( masterLabels, filename )


execute_train()