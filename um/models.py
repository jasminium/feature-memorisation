import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Input, Flatten, Dropout
from tensorflow.keras.layers import concatenate, Activation
from tensorflow.keras.models import Model
from keras.applications.densenet import DenseNet121


# augmentation for mnist
aug = keras.Sequential([
    layers.RandomContrast(0.2),
    layers.RandomCrop(27, 27),
    layers.Resizing(28, 28),
])

# augmentation for cifar
aug_32_32 = keras.Sequential([
    layers.RandomContrast(0.2, input_shape=(32, 32, 3)),
    layers.RandomCrop(31, 31),
    layers.Resizing(32, 32),
    layers.RandomFlip("horizontal"),
    # layers.RandomRotation(0.2)
])



def get_cnn1():

    tf.keras.backend.clear_session()

    input_shape = (32, 32, 3)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation="softmax"),
        ]
    )

    return model

def get_cnn_2():

    tf.keras.backend.clear_session()

    input_shape = (32, 32, 3)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3),
                          padding='same', activation="relu"),
            layers.Conv2D(32, kernel_size=(3, 3),
                          padding='same', activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, kernel_size=(3, 3),
                          padding='same', activation="relu"),
            layers.Conv2D(64, kernel_size=(3, 3),
                          padding='same', activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dense(10, activation="softmax"),
        ]
    )

    return model

def get_cnn_2_bn():

    tf.keras.backend.clear_session()

    input_shape = (32, 32, 3)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3),
                          padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(32, kernel_size=(3, 3),
                          padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3),
                          padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(64, kernel_size=(3, 3),
                          padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(10, activation="softmax"),
        ]
    )

    return model


def get_cnn_2_da():

    tf.keras.backend.clear_session()

    input_shape = (32, 32, 3)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            aug_32_32,
            layers.Conv2D(32, kernel_size=(3, 3),
                          padding='same', activation="relu"),
            layers.Conv2D(32, kernel_size=(3, 3),
                          padding='same', activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, kernel_size=(3, 3),
                          padding='same', activation="relu"),
            layers.Conv2D(64, kernel_size=(3, 3),
                          padding='same', activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dense(10, activation="softmax"),
        ]
    )

    return model


def get_cnn_2_do_da():

    tf.keras.backend.clear_session()

    input_shape = (32, 32, 3)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            aug_32_32,
            layers.Conv2D(32, kernel_size=(3, 3),
                          padding='same', activation="relu"),
            layers.Dropout(0.2),
            layers.Conv2D(32, kernel_size=(3, 3),
                          padding='same', activation="relu"),
            layers.Dropout(0.2),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, kernel_size=(3, 3),
                          padding='same', activation="relu"),
            layers.Dropout(0.2),
            layers.Conv2D(64, kernel_size=(3, 3),
                          padding='same', activation="relu"),
            layers.Dropout(0.2),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )

    return model


def get_cnn_2_do():

    tf.keras.backend.clear_session()

    input_shape = (32, 32, 3)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3),
                          padding='same', activation="relu"),
            layers.Dropout(0.2),
            layers.Conv2D(32, kernel_size=(3, 3),
                          padding='same', activation="relu"),
            layers.Dropout(0.2),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, kernel_size=(3, 3),
                          padding='same', activation="relu"),
            layers.Dropout(0.2),
            layers.Conv2D(64, kernel_size=(3, 3),
                          padding='same', activation="relu"),
            layers.Dropout(0.2),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )

    return model



def get_mlp():

    tf.keras.backend.clear_session()

    input_shape = (28, 28, 1)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            Dense(10, activation='softmax')
        ]
    )

    return model


def get_mlp_bn():

    tf.keras.backend.clear_session()

    input_shape = (28, 28, 1)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.ReLU(),
            Dense(10, activation='softmax')
        ]
    )

    return model


def get_mlp_aug():

    tf.keras.backend.clear_session()

    input_shape = (28, 28, 1)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            aug,
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            Dense(10, activation='softmax')
        ]
    )

    return model


def get_mlp_dp():

    tf.keras.backend.clear_session()

    input_shape = (28, 28, 1)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            Dense(10, activation='softmax')
        ]
    )

    return model


def get_mlp_dp_aug():

    tf.keras.backend.clear_session()

    input_shape = (28, 28, 1)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            aug,
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            Dense(10, activation='softmax')
        ]
    )

    return model



def get_dense():

    data_augmentation = False
    num_classes = 10
    num_dense_blocks = 3
    use_max_pool = False
    input_shape = (32, 32, 3)

    # DenseNet-BC with dataset augmentation
    # Growth rate   | Depth |  Accuracy (paper)| Accuracy (this)      |
    # 12            | 100   |  95.49%          | 93.74%               |
    # 24            | 250   |  96.38%          | requires big mem GPU |
    # 40            | 190   |  96.54%          | requires big mem GPU |
    growth_rate = 12
    depth = 100
    num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)

    num_filters_bef_dense_block = 2 * growth_rate
    compression_factor = 0.5

    # start model definition
    # densenet CNNs (composite function) are made of BN-ReLU-Conv2D
    inputs = Input(shape=input_shape)
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(num_filters_bef_dense_block,
               kernel_size=3,
               padding='same',
               kernel_initializer='he_normal')(x)
    x = concatenate([inputs, x])

    # stack of dense blocks bridged by transition layers
    for i in range(num_dense_blocks):
        # a dense block is a stack of bottleneck layers
        for j in range(num_bottleneck_layers):
            y = BatchNormalization()(x)
            y = Activation('relu')(y)
            y = Conv2D(4 * growth_rate,
                       kernel_size=1,
                       padding='same',
                       kernel_initializer='he_normal')(y)
            if not data_augmentation:
                y = Dropout(0.2)(y)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv2D(growth_rate,
                       kernel_size=3,
                       padding='same',
                       kernel_initializer='he_normal')(y)
            if not data_augmentation:
                y = Dropout(0.2)(y)
            x = concatenate([x, y])

        # no transition layer after the last dense block
        if i == num_dense_blocks - 1:
            continue

        # transition layer compresses num of feature maps and reduces the size by 2
        num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
        num_filters_bef_dense_block = int(
            num_filters_bef_dense_block * compression_factor)
        y = BatchNormalization()(x)
        y = Conv2D(num_filters_bef_dense_block,
                   kernel_size=1,
                   padding='same',
                   kernel_initializer='he_normal')(y)
        if not data_augmentation:
            y = Dropout(0.2)(y)
        x = AveragePooling2D()(y)

    # add classifier on top
    # after average pooling, size of feature map is 1 x 1
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    kernel_initializer='he_normal',
                    activation='softmax')(y)

    # instantiate and compile model
    # orig paper uses SGD but RMSprop works better for DenseNet
    model = Model(inputs=inputs, outputs=outputs)

    return model


def get_densenet121(input_shape=None, preprocessing=None, n_outputs=None, activation='sigmoid', weights='imagenet'):
    
    # train only the classifier layer if we are using pretrained weights
    if weights is None:
        freeze_cnn = False
    else:
        freeze_cnn = True

    base_model = DenseNet121(include_top=False, input_shape=input_shape, weights=weights)

    if freeze_cnn:
        for layer in base_model.layers:
            layer.trainable = False

    model = tf.keras.models.Sequential()
    
    if preprocessing is not None:
        model.add(preprocessing)
    
    layers = [
        base_model,
        # Add two layer classifier
        keras.layers.GlobalAveragePooling2D(input_shape=(1024, 1, 1)),
        # Add a flattern layer 
        Dense(2048, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        # Add a fully-connected layer
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(n_outputs, activation=activation)
    ]

    for lay in layers:
        model.add(lay)

    return model

def get_densenet121_chexpert():
    input_shape = (224, 224, 3)

    seq = tf.keras.Sequential([
        tf.keras.layers.Normalization(mean=0.5330, variance=0.0349, input_shape=input_shape),
        #layers.RandomContrast(0.2),
        #layers.RandomCrop(31, 31),
        layers.RandomFlip("horizontal"),
    ])

    model = get_densenet121(input_shape=input_shape, preprocessing=seq, n_outputs=1, activation='sigmoid')
    return model

def get_densenet121_caltech101():
    input_shape = (224, 224, 3)
    seq = None
    model = get_densenet121(input_shape=input_shape, preprocessing=seq, n_outputs=102, activation='softmax', weights=None)
    return model

def get_densenet121_celeba_hair_color():
    input_shape = (224, 224, 3)
    seq = None
    model = get_densenet121(input_shape=input_shape, preprocessing=seq, n_outputs=4, activation='softmax', weights=None)
    return model

def get_densenet121_celeba_hair_color_aug():
    input_shape = (224, 224, 3)
    seq = tf.keras.Sequential([
        layers.RandomContrast(0.1, input_shape=input_shape),
        layers.RandomRotation(0.1),
        layers.RandomCrop(223, 223),
        layers.RandomFlip("horizontal"),
        layers.Resizing(224, 224),
    ])
    model = get_densenet121(input_shape=input_shape, preprocessing=seq, n_outputs=4, activation='softmax', weights=None)
    return model

def get_densenet121_celeba_hair_color_pretrained():
    input_shape = (224, 224, 3)
    seq = None
    model = get_densenet121(input_shape=input_shape, preprocessing=seq, n_outputs=4, activation='softmax', weights='imagenet')
    return model

def get_densenet121_celeba_hair_color_pretrained_aug():
    input_shape = (224, 224, 3)
    seq = tf.keras.Sequential([
        layers.RandomContrast(0.1, input_shape=input_shape),
        layers.RandomRotation(0.1),
        layers.RandomCrop(223, 223),
        layers.RandomFlip("horizontal"),
        layers.Resizing(224, 224),
    ])
    model = get_densenet121(input_shape=input_shape, preprocessing=seq, n_outputs=4, activation='softmax', weights='imagenet')
    return model

def get_densenet121_caltech101_aug():
    input_shape = (224, 224, 3)
    seq = tf.keras.Sequential([
        layers.RandomContrast(0.1, input_shape=input_shape),
        layers.RandomRotation(0.1),
        layers.RandomCrop(223, 223),
        layers.RandomFlip("horizontal"),
        layers.Resizing(224, 224),
    ])
    model = get_densenet121(input_shape=input_shape, preprocessing=seq, n_outputs=102, activation='softmax', weights=None)
    return model

def get_densenet121_caltech101_pretrained():
    input_shape = (224, 224, 3)
    seq = None
    model = get_densenet121(input_shape=input_shape, preprocessing=seq, n_outputs=102, activation='softmax', weights='imagenet')
    return model

def get_densenet121_caltech101_pretrained_aug():
    input_shape = (224, 224, 3)

    seq = tf.keras.Sequential([
        layers.RandomContrast(0.2, input_shape=input_shape),
        layers.RandomCrop(223, 223),
                
        layers.RandomFlip("horizontal"),
        layers.Resizing(224, 224),
    ])

    model = get_densenet121(input_shape=input_shape, preprocessing=seq, n_outputs=102, activation='softmax', weights='imagenet')
    return model
