from keras import backend as K
from keras.layers import Conv2D, Activation, Concatenate, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense, Dropout

def inception_resnet_block(X, scale, block_type, block_idx, activation='relu'):
    if block_type == 'block35':
        branch_0 = Conv2D(32, 1, name='block35/branch_0/'+str(block_idx), padding='same')(X)
        branch_0 = BatchNormalization(axis=3, name='block35/bn_branch_0/'+str(block_idx))(branch_0)
        branch_0 = Activation('relu', name='block35/branch_0/relu/'+str(block_idx))(branch_0)

        branch_1 = Conv2D(32, 1, name='block35/branch_1_1/'+str(block_idx), padding='same')(X)
        branch_1 = BatchNormalization(axis=3, name='block35/bn_branch_1_1/'+str(block_idx))(branch_1)
        branch_1 = Activation('relu', name='block35/branch_1_1/relu/'+str(block_idx))(branch_1)

        branch_1 = Conv2D(32, 3, name='block35/branch_1_2/'+str(block_idx), padding='same')(branch_1)
        branch_1 = BatchNormalization(axis=3, name='block35/bn_branch_1_2/'+str(block_idx))(branch_1)
        branch_1 = Activation('relu', name='block35/branch_1_2/relu/'+str(block_idx))(branch_1)

        branch_2 = Conv2D(32, 1, name='block35/branch_2_1/'+str(block_idx), padding='same')(X)
        branch_2 = BatchNormalization(axis=3, name='block35/bn_branch_2_1/'+str(block_idx))(branch_2)
        branch_2 = Activation('relu', name='block35/branch_2_1/relu/'+str(block_idx))(branch_2)

        branch_2 = Conv2D(48, 3, name='block35/branch_2_2/'+str(block_idx), padding='same')(branch_2)
        branch_2 = BatchNormalization(axis=3, name='block35/bn_branch_2_2/'+str(block_idx))(branch_2)
        branch_2 = Activation('relu', name='block35/branch_2_2/relu/'+str(block_idx))(branch_2)

        branch_2 = Conv2D(64, 3, name='block35/branch_2_3/'+str(block_idx), padding='same')(branch_2)
        branch_2 = BatchNormalization(axis=3, name='block35/bn_branch_2_3/'+str(block_idx))(branch_2)
        branch_2 = Activation('relu', name='block35/branch_2_3/relu/'+str(block_idx))(branch_2)

        branches = [branch_0, branch_1, branch_2]

    elif block_type == 'block17':
        branch_0 = Conv2D(192, 1, name='block17/branch_0/'+str(block_idx), padding='same')(X)
        branch_0 = BatchNormalization(axis=3, name='block17/bn_branch_0/'+str(block_idx))(branch_0)
        branch_0 = Activation('relu', name='block17/branch_0/relu/'+str(block_idx))(branch_0)

        branch_1 = Conv2D(128, 1, name='block17/branch_1_1/'+str(block_idx), padding='same')(X)
        branch_1 = BatchNormalization(axis=3, name='block17/bn_branch_1_1/'+str(block_idx))(branch_1)
        branch_1 = Activation('relu', name='block17/branch_1_1/relu/'+str(block_idx))(branch_1)

        branch_1 = Conv2D(160, [1, 7], name='block17/branch_1_2/'+str(block_idx), padding='same')(branch_1)
        branch_1 = BatchNormalization(axis=3, name='block17/bn_branch_1_2/'+str(block_idx))(branch_1)
        branch_1 = Activation('relu', name='block17/branch_1_2/relu/'+str(block_idx))(branch_1)

        branch_1 = Conv2D(192, [7, 1], name='block17/branch_1_3/'+str(block_idx), padding='same')(branch_1)
        branch_1 = BatchNormalization(axis=3, name='block17/bn_branch_1_3/'+str(block_idx))(branch_1)
        branch_1 = Activation('relu', name='block17/branch_1_3/relu/'+str(block_idx))(branch_1)
        branches = [branch_0, branch_1]

    elif block_type == 'block8':

        branch_0 = Conv2D(192, 1, name='block8/branch_0/'+str(block_idx), padding='same')(X)
        branch_0 = BatchNormalization(axis=3, name='block8/bn_branch_0/'+str(block_idx))(branch_0)
        branch_0 = Activation('relu', name='block8/branch_0/relu/'+str(block_idx))(branch_0)

        branch_1 = Conv2D(192, 1, name='block8/branch_1_1/'+str(block_idx), padding='same')(X)
        branch_1 = BatchNormalization(axis=3, name='block8/bn_branch_1_1/'+str(block_idx))(branch_1)
        branch_1 = Activation('relu', name='block8/branch_1_1/relu/'+str(block_idx))(branch_1)

        branch_1 = Conv2D(224, [1, 3], name='block8/branch_1_2/'+str(block_idx), padding='same')(branch_1)
        branch_1 = BatchNormalization(axis=3, name='block8/bn_branch_1_2/'+str(block_idx))(branch_1)
        branch_1 = Activation('relu', name='block8/branch_1_2/relu/'+str(block_idx))(branch_1)

        branch_1 = Conv2D(256, [3, 1], name='block8/branch_1_3/'+str(block_idx), padding='same')(branch_1)
        branch_1 = BatchNormalization(axis=3, name='block8/bn_branch_1_3/'+str(block_idx))(branch_1)
        branch_1 = Activation('relu', name='block8/branch_1_3/relu/'+str(block_idx))(branch_1)
        branches = [branch_0, branch_1]
    else:
        raise ValueError('unknown block type')

    block_name = block_type + '_' + str(block_idx)
    mixed = Concatenate(axis=3, name=block_name + '_mixed')(branches)
    up = Conv2D(K.int_shape(X)[3], 1, name=block_name + '_conv', padding='same')(mixed)

    X = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale, output_shape=K.int_shape(X)[1:],arguments={'scale': scale}, name=block_name+"_Lambda")([X, up])
    if activation is not None:
        X = Activation(activation, name=block_name + '_ac')(X)

    return X



def conv2d_bn(x,filters,kernel_size,strides=1,padding='same',activation='relu',use_bias=False,name=None):
    x = Conv2D(filters,kernel_size,strides=strides,padding=padding,use_bias=use_bias,name=name)(x)
    if not use_bias:
        bn_axis = 3
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def InceptionResnetV2(X_input):
    #X_input = Input(input_shape)
    #print( X_input )
    # 299 X 299 X 3  ->   # 149 X 149 X 32
    X = Cropping2D(cropping=((11, 11), (11, 11)))(X_input)
    X = Conv2D(32, 3, strides=2, name='conv1', padding='valid')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu', name="arelu1")(X)
    # 149 X 149 X 32   ->  # 147 x 147 X 32
    X = Conv2D(32, 3, name='conv2', padding='valid')(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu', name="arelu2")(X)
    # 147 x 147 X 32   ->    # 147 X 147 X 64
    X = Conv2D(64, 3, name='conv3', padding='same')(X)
    X = BatchNormalization(axis=3, name='bn3')(X)
    X = Activation('relu',name="arelu3")(X)
    # 147 X 147 X 64   ->    # 73 X 73 X 64
    X = MaxPooling2D(3, strides=2, name="mp1")(X)
    # 73 X 73 X 64    ->    # 73 X 73 X 80
    X = Conv2D(80, 1, name='conv4', padding='valid')(X)
    X = BatchNormalization(axis=3, name='bn4')(X)
    X = Activation('relu', name="arelu4")(X)
    # 73 X 73 X 80    ->    # 71 X 71 X 192
    X = Conv2D(192, 3, name='conv5', padding='valid')(X)
    X = BatchNormalization(axis=3, name='bn5')(X)
    X = Activation('relu', name="arelu5")(X)
    # 71 X 71 X 192  ->  # 35 X 35 X 192
    X = MaxPooling2D(3, strides=2, name="mp2")(X)
    # 35 X 35 X 192 -> 35 X 35 X 96
    branch_0 = Conv2D(96, 1, name='mixed_5b/branch_0', padding='same')(X)
    branch_0 = BatchNormalization(axis=3, name='mixed_5b/bn_branch_0')(branch_0)
    branch_0 = Activation('relu',name="arelu6")(branch_0)

    # 35 X 35 X 192 -> 35 X 35 X 64
    branch_1 = Conv2D(48, 1, name='mixed_5b/branch_1_1', padding='same')(X)
    branch_1 = BatchNormalization(axis=3, name='mixed_5b/bn_branch_1_1')(branch_1)
    branch_1 = Activation('relu',name="arelu7")(branch_1)

    branch_1 = Conv2D(64, 5, name='mixed_5b/branch_1_2', padding='same')(branch_1)
    branch_1 = BatchNormalization(axis=3, name='mixed_5b/bn_branch_1_2')(branch_1)
    branch_1 = Activation('relu',name="arelu8")(branch_1)

    # 35 X 35 X 192 -> 35 X 35 X 96
    branch_2 = Conv2D(64, 1, name='mixed_5b/branch_2_1', padding='same')(X)
    branch_2 = BatchNormalization(axis=3, name='mixed_5b/bn_branch_2_1')(branch_2)
    branch_2 = Activation('relu',name="arelu9")(branch_2)

    branch_2 = Conv2D(96, 3, name='mixed_5b/branch_2_2', padding='same')(branch_2)
    branch_2 = BatchNormalization(axis=3, name='mixed_5b/bn_branch_2_2')(branch_2)
    branch_2 = Activation('relu',name="arelu10")(branch_2)

    branch_2 = Conv2D(96, 3, name='mixed_5b/branch_2_3', padding='same')(branch_2)
    branch_2 = BatchNormalization(axis=3, name='mixed_5b/bn_branch_2_3')(branch_2)
    branch_2 = Activation('relu',name="arelu11")(branch_2)

    # 35 X 35 X 192 -> 35 X 35 X 64
    branch_pool = AveragePooling2D(3, strides=1, padding='same',name="ap1")(X)
    branch_pool = Conv2D(64, 1, name='mixed_5b/branch_pool_1', padding='same')(branch_pool)
    branch_pool = BatchNormalization(axis=3, name='mixed_5b/bn_branch_pool_1')(branch_pool)
    branch_pool = Activation('relu',name="arelu12")(branch_pool)

    branches = [branch_0, branch_1, branch_2, branch_pool]
    X = Concatenate(axis=3, name='mixed_5b')(branches)  # 35 X 35 X 320

    # 10x block35
    #for block_idx in range(1, 11):
    #    X = inception_resnet_block(X, scale=0.17, block_type='block35', block_idx=block_idx)

    branch_0 = Conv2D(384, 3, strides=2, name='mixed_6a/branch_0', padding='valid')(X)
    branch_0 = BatchNormalization(axis=3, name='mixed_6a/bn_branch_0')(branch_0)
    branch_0 = Activation('relu',name="arelu13")(branch_0)

    branch_1 = Conv2D(256, 1, name='mixed_6a/branch_1_1', padding='same')(X)
    branch_1 = BatchNormalization(axis=3, name='mixed_6a/bn_branch_1_1')(branch_1)
    branch_1 = Activation('relu',name="arelu14")(branch_1)

    branch_1 = Conv2D(256, 3, name='mixed_6a/branch_1_2', padding='same')(branch_1)
    branch_1 = BatchNormalization(axis=3, name='mixed_6a/bn_branch_1_2')(branch_1)
    branch_1 = Activation('relu',name="arelu15")(branch_1)

    branch_1 = Conv2D(384, 3, strides=2, name='mixed_6a/branch_1_3', padding='valid')(branch_1)
    branch_1 = BatchNormalization(axis=3, name='mixed_6a/bn_branch_1_3')(branch_1)
    branch_1 = Activation('relu',name="arelu16")(branch_1)

    branch_pool = MaxPooling2D(3, strides=2, padding='valid', name="mp4")(X)
    branches = [branch_0, branch_1, branch_pool]
    X = Concatenate(axis=3, name='mixed_6a')(branches)

    #for block_idx in range(1, 21):
    #    X = inception_resnet_block(X, scale=0.1, block_type='block17', block_idx=block_idx)

    branch_0 = Conv2D(256, 1, name='mixed_7a/branch_0', padding='same')(X)
    branch_0 = BatchNormalization(axis=3, name='mixed_7a/bn_branch_0')(branch_0)
    branch_0 = Activation('relu',name="arelu17")(branch_0)

    branch_0 = Conv2D(384, 3, strides=2, name='mixed_7a/branch_0_1', padding='valid')(branch_0)
    branch_0 = BatchNormalization(axis=3, name='mixed_7a/bn_branch_0_1')(branch_0)
    branch_0 = Activation('relu',name="arelu18")(branch_0)

    branch_1 = Conv2D(256, 1, name='mixed_7a/branch_1_1', padding='same')(X)
    branch_1 = BatchNormalization(axis=3, name='mixed_7a/bn_branch_1_1')(branch_1)
    branch_1 = Activation('relu',name="arelu19")(branch_1)

    branch_1 = Conv2D(288, 3, strides=2, name='mixed_7a/branch_1_2', padding='valid')(branch_1)
    branch_1 = BatchNormalization(axis=3, name='mixed_7a/bn_branch_1_2')(branch_1)
    branch_1 = Activation('relu',name="arelu20")(branch_1)

    branch_2 = Conv2D(256, 1, name='mixed_7a/branch_2_1', padding='same')(X)
    branch_2 = BatchNormalization(axis=3, name='mixed_7a/bn_branch_2_1')(branch_2)
    branch_2 = Activation('relu',name="arelu21")(branch_2)

    branch_2 = Conv2D(288, 3, name='mixed_7a/branch_2_2', padding='same')(branch_2)
    branch_2 = BatchNormalization(axis=3, name='mixed_7a/bn_branch_2_2')(branch_2)
    branch_2 = Activation('relu',name="arelu22")(branch_2)

    branch_2 = Conv2D(320, 3, strides=2, name='mixed_7a/branch_2_3', padding='valid')(branch_2)
    branch_2 = BatchNormalization(axis=3, name='mixed_7a/bn_branch_2_3')(branch_2)
    branch_2 = Activation('relu',name="arelu23")(branch_2)

    branch_pool = MaxPooling2D(3, strides=2, padding='valid',name="mp5")(X)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    X = Concatenate(axis=3, name='mixed_7a')(branches)

    #for block_idx in range(1, 10):
    #    X = inception_resnet_block(X, scale=0.2, block_type='block8', block_idx=block_idx)

    X = inception_resnet_block(X, scale=1., activation=None, block_type='block8', block_idx=10)

    X = Conv2D(1536, 1, name='conv_7b', padding='same')(X)
    X = BatchNormalization(axis=3, name='bn7b')(X)
    X = Activation('relu',name="arelu24")(X)
    X = AveragePooling2D(K.int_shape(X)[1:3], strides=1, padding='valid', name="ap3")(X)
    X = Flatten()(X)
    X = Dropout(1.0)(X)
    X = Dense(128,name='embeddings' )(X)
    X = Lambda( lambda x: K.l2_normalize(x, axis=1), name="LambdaForNormalization")(X)
    #model = Model( inputs = X_input, outputs=X, name="FaceRecoModel")
    return X

