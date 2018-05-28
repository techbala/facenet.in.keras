
from keras.models import model_from_json
from tensorflow import Graph, Session
import random
from sklearn.model_selection import KFold
from scipy import interpolate
from util import *
#from inceptionresnet import InceptionResnetV2

def testPairs( datasets, nrof_pairs):
    step = 0
    pairs = []
    issame = []
    while step < nrof_pairs:
        imageClass = random.choice(datasets)
        paths = imageClass.image_paths
        first = random.choice( paths )
        second = random.choice( paths )
        if first != second :
            pairs.append(first)
            pairs.append(second)
            issame.append(True)
            step += 1

    step = 0
    while step < nrof_pairs:
        first = random.choice(datasets)
        second = random.choice(datasets)
        if first != second:
            pairs.append(random.choice(first.image_paths))
            pairs.append(random.choice(second.image_paths))
            issame.append(False)
            step += 1
    return pairs, issame

def get_eval_embeddings( eval_datasets ):
    outer_graph = Graph()
    with outer_graph.as_default():
        outer_session = Session()
        with outer_session.as_default():
            #network = load_model(parent_directory + model_directory + complete_model, custom_objects={'triplet_loss':train.triplet_loss} )

            with open(parent_directory + model_directory + model_architecture_as_json, "r") as json_file:
                model_json = json_file.read()
            network = model_from_json( model_json )
            print( network.summary() )
            network.load_weights( parent_directory + model_directory + weights_of_model )
            embeds = network.predict(np.stack(getImages(eval_datasets)))
    return embeds

def evaluate( eval_datasets, issame, nrof_folds=10 ):
    embeddings = get_eval_embeddings( eval_datasets )
    print( embeddings.size )
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,np.asarray(issame), nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,np.asarray(issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    print( actual_issame )
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    print( "n pairs %d and n thresholds %d" %( nrof_pairs, nrof_thresholds ) )
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    diff = np.subtract(embeddings1, embeddings2)
    print( 'diff between embeddings is ' )
    print( diff )
    dist = np.sum(np.square(diff), 1)
    print( dist )
    indices = np.arange(nrof_pairs)
    print( indices )

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        print( 'accuracy loop %d' % (fold_idx ))
        print( train_set )
        print( test_set )
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        print( 'best threshold index %d' % ( best_threshold_index ) )
        for threshold_idx, threshold in enumerate(thresholds):
            print( 'threshold idx %d and threshold %f' %( threshold_idx, threshold ) )
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,dist[test_set],actual_issame[test_set])
        print( tprs )
        print( fprs )
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],actual_issame[test_set])
        print( accuracy )

    print( "################ Final TPR and FPR are #######################")
    tpr = np.mean(tprs, 0)
    print(  tpr )
    fpr = np.mean(fprs, 0)
    print( fpr )
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    print( ' in accuracy ')
    print( ' threshold %f ' %( threshold ))
    print( dist )
    print( actual_issame )
    predict_issame = np.less(dist, threshold)
    print( predict_issame )
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    print( ' true positive %f' %( tp ) )
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    print(' false positive %f' % (fp))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    print(' true negative %f' % (tn))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    print(' false negative %f' % (fn))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    print(' tpt %f' % (tpr))
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    print(' fpr %f' % (fpr))
    acc = float(tp + tn) / dist.size
    print(' acc %f' % (acc))
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        print('val loop %d' % (fold_idx))
        print(train_set)
        print(test_set)
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        print( far_train )
        print( far_target )
        if np.max(far_train) >= far_target:
            print( ' far train is greater than far target')
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            print( f )
            threshold = f(far_target)
            print( threshold )
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

def calculate_val_far(threshold, dist, actual_issame):
    print(' in val far ')
    print(' threshold %f ' % (threshold))
    print(dist)
    print(actual_issame)
    predict_issame = np.less(dist, threshold)
    print(predict_issame)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    print(' true_accept %f' % (true_accept))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    print(' false_accept %f' % (false_accept))
    n_same = np.sum(actual_issame)
    print( ' n_same %f' %(n_same))
    n_diff = np.sum(np.logical_not(actual_issame))
    print(' n_diff %f' % (n_diff))
    val = float(true_accept) / float(n_same)
    print(' val %f' % (val))
    far = float(false_accept) / float(n_diff)
    print(' far %f' % (far))
    return val, far

def execute_evaluate():
    datasets = get_dataset( parent_directory + evaluate_directory )
    print( datasets )
    pairs, issame = testPairs(datasets, 100)
    print( pairs )
    print( issame )
    evaluate(np.stack(pairs), issame)

execute_evaluate()