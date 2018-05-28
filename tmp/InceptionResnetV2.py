from keras import backend as K
from keras.layers import Conv2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense, Dropout


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu')
    if block_type == 'block35':
        branch_0 = Conv2D(32, 1, name='branch_0', padding='same')(X)
        branch_0 = BatchNormalization(axis=3, name='bn.branch_0')(branch_0)
        branch_0 = Activation('relu')(branch_0)

        branch_1 = Conv2D(32, 1, name='branch_1_1', padding='same')(X)
        branch_1 = BatchNormalization(axis=3, name='bn.branch_1.1')(branch_1)
        branch_1 = Activation('relu')(branch_1)

        branch_1 = Conv2D(32, 3, name='branch_1_2', padding='same')(branch_1)
        branch_1 = BatchNormalization(axis=3, name='bn.branch_1.2')(branch_1)
        branch_1 = Activation('relu')(branch_1)

        branch_2 = Conv2D(32, 1, name='branch_2_1', padding='same')(X)
        branch_2 = BatchNormalization(axis=3, name='bn.branch_2.1')(branch_2)
        branch_2 = Activation('relu')(branch_2)

        branch_2 = Conv2D(48, 3, name='branch_2_2', padding='same')(branch_2)
        branch_2 = BatchNormalization(axis=3, name='bn.branch_2.3')(branch_2)
        branch_2 = Activation('relu')(branch_2)

        branch_2 = Conv2D(64, 3, name='branch_2_3', padding='same')(branch_2)
        branch_2 = BatchNormalization(axis=3, name='bn.branch_2.3')(branch_2)
        branch_2 = Activation('relu')(branch_2)

        branches = [branch_0, branch_1, branch_2]

    elif block_type == 'block17':
        branch_0 = Conv2D(192, 1, name='branch_0', padding='same')(X)
        branch_0 = BatchNormalization(axis=3, name='bn.branch_0')(branch_0)
        branch_0 = Activation('relu')(branch_0)

        branch_1 = Conv2D(128, 1, name='branch_1_1', padding='same')(X)
        branch_1 = BatchNormalization(axis=3, name='bn.branch_1.1')(branch_1)
        branch_1 = Activation('relu')(branch_1)

        branch_1 = Conv2D(160, [1, 7], name='branch_1_2', padding='same')(branch_1)
        branch_1 = BatchNormalization(axis=3, name='bn.branch_1.2')(branch_1)
        branch_1 = Activation('relu')(branch_1)

        branch_1 = Conv2D(192, [7, 1], name='branch_1_2', padding='same')(branch_1)
        branch_1 = BatchNormalization(axis=3, name='bn.branch_1.2')(branch_1)
        branch_1 = Activation('relu')(branch_1)
        branches = [branch_0, branch_1]

    elif block_type == 'block8':

        branch_0 = Conv2D(192, 1, name='branch_0', padding='same')(X)
        branch_0 = BatchNormalization(axis=3, name='bn.branch_0')(branch_0)
        branch_0 = Activation('relu')(branch_0)

        branch_1 = Conv2D(192, 1, name='branch_1_1', padding='same')(X)
        branch_1 = BatchNormalization(axis=3, name='bn.branch_1.1')(branch_1)
        branch_1 = Activation('relu')(branch_1)

        branch_1 = Conv2D(224, [1, 3], name='branch_1_2', padding='same')(branch_1)
        branch_1 = BatchNormalization(axis=3, name='bn.branch_1.2')(branch_1)
        branch_1 = Activation('relu')(branch_1)

        branch_1 = Conv2D(256, [3, 1], name='branch_1_2', padding='same')(branch_1)
        branch_1 = BatchNormalization(axis=3, name='bn.branch_1.2')(branch_1)
        branch_1 = Activation('relu')(branch_1)
        branches = [branch_0, branch_1]
    else:
        raise ValueError('unknown block type')

    block_name = block_type + '_' + str(block_idx)
    mixed = Concatenate(axis=3, name=block_name + '_mixed')(branches)
    up = Conv2D(K.int_shape(x)[channel_axis], 1, name=block_name + '_conv', padding='same')(mixed)

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale, output_shape=K.int_shape(x)[1:],arguments={'scale': scale}, name=block_name)([x, up])
    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(x)

    return x


def InceptionResnetV2(input_shape):
    X = Input(input_shape)
    # 299 X 299 X 3  ->   # 149 X 149 X 32
    X = Conv2D(32, 3, strides=2, name='conv1', padding='valid')(X)

    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)
    # 149 X 149 X 32   ->  # 147 x 147 X 32
    X = Conv2D(32, 3, name='conv2', padding='valid')(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X)
    # 147 x 147 X 32   ->    # 147 X 147 X 64
    X = Conv2D(64, 3, name='conv3', padding='same')(X)
    X = BatchNormalization(axis=3, name='bn3')(X)
    X = Activation('relu')(X)
    # 147 X 147 X 64   ->    # 73 X 73 X 64
    X = MaxPooling2D(3, strides=2)(X)
    # 73 X 73 X 64    ->    # 73 X 73 X 80
    X = Conv2D(80, 1, name='conv4', padding='valid')(X)
    X = BatchNormalization(axis=3, name='bn4')(X)
    X = Activation('relu')(X)
    # 73 X 73 X 80    ->    # 71 X 71 X 192
    X = Conv2D(192, 3, name='conv5', padding='valid')(X)
    X = BatchNormalization(axis=3, name='bn5')(X)
    X = Activation('relu')(X)
    # 71 X 71 X 192  ->  # 35 X 35 X 192
    X = MaxPooling2D(3, strides=2)(X)

    # 35 X 35 X 192 -> 35 X 35 X 96
    branch_0 = Conv2D(96, 1, name='branch_0', padding='same')(X)
    branch_0 = BatchNormalization(axis=3, name='bn.branch_0')(branch_0)
    branch_0 = Activation('relu')(branch_0)

    # 35 X 35 X 192 -> 35 X 35 X 64
    branch_1 = Conv2D(48, 1, name='branch_1_1', padding='same')(X)
    branch_1 = BatchNormalization(axis=3, name='bn.branch_1.1')(branch_1)
    branch_1 = Activation('relu')(branch_1)

    branch_1 = Conv2D(64, 5, name='branch_1_2', padding='same')(branch_1)
    branch_1 = BatchNormalization(axis=3, name='bn.branch_1.2')(branch_1)
    branch_1 = Activation('relu')(branch_1)

    # 35 X 35 X 192 -> 35 X 35 X 96
    branch_2 = Conv2D(64, 1, name='branch_2_1', padding='same')(X)
    branch_2 = BatchNormalization(axis=3, name='bn.branch_2.1')(branch_2)
    branch_2 = Activation('relu')(branch_2)

    branch_2 = Conv2D(96, 3, name='branch_2_2', padding='same')(branch_2)
    branch_2 = BatchNormalization(axis=3, name='bn.branch_2.3')(branch_2)
    branch_2 = Activation('relu')(branch_2)

    branch_2 = Conv2D(96, 3, name='branch_2_3', padding='same')(branch_2)
    branch_2 = BatchNormalization(axis=3, name='bn.branch_2.3')(branch_2)
    branch_2 = Activation('relu')(branch_2)

    # 35 X 35 X 192 -> 35 X 35 X 64
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(X)
    branch_pool = Conv2D(64, 1, name='branch_pool_1', padding='same')(branch_pool)
    branch_pool = BatchNormalization(axis=3, name='bn.branch_pool_1')(branch_pool)
    branch_pool = Activation('relu')(branch_pool)

    branches = [branch_0, branch_1, branch_2, branch_pool]
    X = concatenate(axis=3, name='mixed_5b')(branches)  # 35 X 35 X 320

    # 10x block35
    for block_idx in range(1, 11):
        X = inception_resnet_block(X, scale=0.17, block_type='block35', block_idx=block_idx)

    branch_0 = Conv2D(384, 3, strides=2, name='branch_0', padding='valid')(X)
    branch_0 = BatchNormalization(axis=3, name='bn.branch_0')(branch_0)
    branch_0 = Activation('relu')(branch_0)

    branch_1 = Conv2D(256, 1, name='branch_1_1', padding='same')(X)
    branch_1 = BatchNormalization(axis=3, name='bn.branch_1.1')(branch_1)
    branch_1 = Activation('relu')(branch_1)

    branch_1 = Conv2D(256, 3, name='branch_1_2', padding='same')(branch_1)
    branch_1 = BatchNormalization(axis=3, name='bn.branch_1.2')(branch_1)
    branch_1 = Activation('relu')(branch_1)

    branch_1 = Conv2D(384, 3, strides=2, name='branch_1_2', padding='valid')(branch_1)
    branch_1 = BatchNormalization(axis=3, name='bn.branch_1.2')(branch_1)
    branch_1 = Activation('relu')(branch_1)

    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(X)
    branches = [branch_0, branch_1, branch_pool]
    X = concatenate(axis=3, name='mixed_6a')(branches)

    for block_idx in range(1, 21):
        X = inception_resnet_block(X, scale=0.1, block_type='block17', block_idx=block_idx)

    branch_0 = Conv2D(256, 1, name='branch_0', padding='same')(X)
    branch_0 = BatchNormalization(axis=3, name='bn.branch_0')(branch_0)
    branch_0 = Activation('relu')(branch_0)

    branch_0 = Conv2D(384, 3, strides=2, name='branch_0', padding='valid')(branch_0)
    branch_0 = BatchNormalization(axis=3, name='bn.branch_0')(branch_0)
    branch_0 = Activation('relu')(branch_0)

    branch_1 = Conv2D(256, 1, name='branch_1_1', padding='same')(X)
    branch_1 = BatchNormalization(axis=3, name='bn.branch_1.1')(branch_1)
    branch_1 = Activation('relu')(branch_1)

    branch_1 = Conv2D(288, 3, strides=2, name='branch_1_2', padding='valid')(branch_1)
    branch_1 = BatchNormalization(axis=3, name='bn.branch_1.2')(branch_1)
    branch_1 = Activation('relu')(branch_1)

    branch_2 = Conv2D(256, 1, name='branch_2_1', padding='same')(X)
    branch_2 = BatchNormalization(axis=3, name='bn.branch_2.1')(branch_2)
    branch_2 = Activation('relu')(branch_2)

    branch_2 = Conv2D(288, 3, name='branch_2_2', padding='same')(branch_2)
    branch_2 = BatchNormalization(axis=3, name='bn.branch_2.3')(branch_2)
    branch_2 = Activation('relu')(branch_2)

    branch_2 = Conv2D(320, 3, strides=2, name='branch_2_3', padding='valid')(branch_2)
    branch_2 = BatchNormalization(axis=3, name='bn.branch_2.3')(branch_2)
    branch_2 = Activation('relu')(branch_2)

    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(X)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    X = concatenate(axis=3, name='mixed_7a')(branches)

    for block_idx in range(1, 10):
        X = inception_resnet_block(X, scale=0.2, block_type='block8', block_idx=block_idx)

    X = inception_resnet_block(X, scale=1., activation=None, block_type='block8', block_idx=10)

    X = Conv2D(1536, 1, name='conv_7b', padding='same')(X)
    X = BatchNormalization(axis=3, name='bn7b')(X)
    X = Activation('relu')(X)

    X = AveragePooling2D(K.int_shape(X)[1:3], strides=1, padding='valid')(X)
    X = Flatten()(X)
    X = Dropout(0.8)(X)
    X = Dense(128)(X)

    return X


 X = InceptionResnetV2((299, 299, 3))
 model = Model(None, X, name='inception_resnet_v2')
 model.summary()