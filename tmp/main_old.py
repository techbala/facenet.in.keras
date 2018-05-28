import tensorflow as tf
import numpy as np
import os
import h5py
import errno
from numpy import genfromtxt
from keras import backend as K
from keras.models import Model
from keras.preprocessing import image
from inceptionresnet import InceptionResnetV2


height = 321
width = 321
people_per_batch = 15
images_per_person = 10
max_nrof_epochs = 500
epoch_size = 1000

dataset_directory = '/ai/sources/keras.ins.facenet'
input_directory ='/lfw.face.aligned.with.321'
embed_directory = '/lfw.embed.with.128'
datasets = get_dataset( dataset_directory + input_directory )

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
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

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
			
def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    for i in xrange(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' % 
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
    while len(image_paths)<nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images-len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index]*nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i+=1
  
    return image_paths, num_per_class
	
def saveEmbeddings( height, width, datasets, inputfolder, embedfolder ):
	input_shape = (height,width, 3)
	X_input = Input( input_shape )
	X = InceptionResnetV2(X_input)
	model = Model( inputs = X_input, outputs = X)
	
	for folder in datasets:
    for path in folder.image_paths:
		img = image.load_img(path, target_size=(height, width))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
        filepath = path.replace(".png", ".txt")
		filepath = filepath.replace(input_directory, embed_directory)
		createFolder(filepath)
		np.savetxt(filepath, model.predict(x))
		

	
		
	




#saveEmbeddings( height, width, datasets, input_directory, embed_directory)
image_paths, num_per_class = sample_people( datasets, people_per_batch, images_per_person )
print( image_paths )
print( num_per_class )

epoch = 0
step = 0
while epoch < max_nrof_epochs:
	step = step + 1
	epoch = step // epoch_size
	train()
	
	





		

		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
####################### Backup of old Code not sure about the order of execution. #######################	
#createFolder(filepath)
#with open( filepath, 'w') as embedfile:
       	#embedfile.write(model.predict(x ))
		#print( model.predict(x) )
 		#print( type( model.predict(x)) )

#img_path = '/ai/sources/ft/lfw.aligned/Tony_Blair/Tony_Blair_0009.png'
#img_path ='/ai/sources/ft/lfw.aligned.small/Tom_Ridge/Tom_Ridge_0010.png'
#img = image.load_img( img_path, target_size=(height,width))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)

#dataset = get_dataset("/ai/sources/ft/lfw.aligned.small/Tiger_Woods")
#for i in dataset:
    #print(i.__str__())

#X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
#X_train = X_train_orig/255.
#X_test = X_test_orig/255.
#Y_train = convert_to_one_hot(Y_train_orig, 6).T
#Y_test = convert_to_one_hot(Y_test_orig, 6).T

#input_shape = (height,width, 3)
#X_input = Input( input_shape )

#X = InceptionResnetV2(X_input)
#print( X )

#model = Model( inputs =X_input, outputs =X )
#model.compile(loss='mse', optimizer='adam')

#embed = Model( inputs = model.input, outputs = model.get_layer('embeddings').output)
#print( model.predict(x))
#model.summary()
####################### Backup of old Code not sure about the order of execution. #######################
