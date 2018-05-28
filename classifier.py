
from keras.models import model_from_json
from tensorflow import Graph, Session
from util import *
from sklearn.svm import SVC
import pickle

def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

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

def get_eval_embeddings( eval_datasets ):
    outer_graph = Graph()
    with outer_graph.as_default():
        outer_session = Session()
        with outer_session.as_default():
            with open(parent_directory + model_directory + model_architecture_as_json, "r") as json_file:
                model_json = json_file.read()
            network = model_from_json( model_json )
            network.load_weights( parent_directory + model_directory + weights_of_model )
            embeds = network.predict(np.stack(getImages(eval_datasets)))
    return embeds

def classify( train_set, test_set):
    train_paths, train_labels = get_image_labels( train_set )
    test_paths, test_labels = get_image_labels( test_set )

    train_embeddings = get_eval_embeddings( train_paths )
    test_embeddings = get_eval_embeddings( test_paths )

    model = SVC(kernel='linear', C=1, gamma = 'auto')
    model.fit( train_embeddings, train_labels )
    model.score( train_embeddings, train_labels )
    predictions = model.predict( test_embeddings )
    matches = [ ( predictions[i], test_labels[i]) for i in range( len(predictions))]
    print( matches )
    matchedCount = len([ x for x, y in matches if x == y ])
    print( matchedCount )
    print( matchedCount / len( test_labels) )

def classify_train( train_set ):
    paths, labels = get_image_paths_and_labels( train_set )
    print( paths )
    print( labels )
    embeddings = get_eval_embeddings( paths )
    model = SVC(kernel='linear', probability=True)
    model.fit(embeddings, labels)
    # Create a list of class names
    class_names = [cls.name.replace('_', ' ') for cls in train_set]
    classifier_filename_exp = parent_directory+model_directory+classifier_model
    # Saving classifier model
    with open( classifier_filename_exp, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    print('Saved classifier model to file "%s"' % classifier_filename_exp)

def classify_test( test_set ):
    paths, labels = get_image_paths_and_labels(test_set)
    print( paths )
    print( labels )
    embeddings = get_eval_embeddings(paths)
    classifier_filename_exp = parent_directory + model_directory + classifier_model

    with open(classifier_filename_exp, 'rb') as infile:
        (model, class_names) = pickle.load(infile)

    print('Loaded classifier model from file "%s"' % classifier_filename_exp)

    predictions = model.predict_proba(embeddings)
    print( predictions.shape )
    best_class_indices = np.argmax(predictions, axis=1)
    print( best_class_indices )
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    print( best_class_probabilities )
    predicted_labels = []
    true_class_names = [cls.name.replace('_', ' ') for cls in test_set]
    for i in range(len(best_class_indices)):
        print('%4d  %s:  %s:  %.3f' % (i, class_names[best_class_indices[i]], true_class_names[i], best_class_probabilities[i]))
        predicted_labels.append( class_names[best_class_indices[i]] )

    accuracy = np.mean(np.equal(best_class_indices, labels))
    print('Accuracy: %.3f' % accuracy)

def execute_classify():
    datasets = get_dataset( parent_directory + evaluate_directory )
    print( datasets )
    train_set, test_set = split_dataset(datasets, 4, 3)
    #classify_train( train_set )
    #classify_test( test_set )
    classify( train_set, test_set )

execute_classify()