import numpy as np
import tensorflow as tf
import artifacters
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
from keras.preprocessing import image as image_pre

from utils import normalise
from utils import show_image

tf.compat.v1.enable_eager_execution()

fmnist_mean = 0.2860405969887955
fmnist_std = 0.35302424451492237
mnist_mean = 0.1307
mnist_std = 0.3081
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)

artifact = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
])
artifact = np.expand_dims(artifact, axis=-1).astype(np.float32)
artifacter = artifacters.Artifacter(
    1, artifact, artifact)

def get_fmnist():
    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = np.expand_dims(train_images, -1)
    train_images = np.copy(train_images)
    train_labels = np.copy(train_labels)
    test_images = np.expand_dims(test_images, -1)
    return (train_images, train_labels, test_images, test_labels)

def get_fmnist_std():
    train_images, train_labels, test_images, test_labels = get_fmnist()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = normalise(train_images, fmnist_mean, fmnist_std)
    test_images = normalise(test_images, fmnist_mean, fmnist_std)
    return (train_images, train_labels, test_images, test_labels)

def get_fmnist_leaky_std():
    train_images, train_labels, test_images, test_labels = get_fmnist()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = artifacter.augment_images(train_images.copy(), artifact)
    test_images = artifacter.augment_images(test_images.copy(), artifact)
    train_images = normalise(train_images, fmnist_mean, fmnist_std)
    test_images = normalise(test_images, fmnist_mean, fmnist_std)
    return (train_images, train_labels, test_images, test_labels)


def get_mnist():
    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
    return (train_images, train_labels, test_images, test_labels)


def get_cifar10():
    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images = np.copy(train_images)
    return (train_images, train_labels, test_images, test_labels)

def get_mnist_3_channel(imsize=(32,32)):
    # cifar (numpy arrays)
    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.mnist.load_data()
    def to_3(train_images):
        train_images = np.stack(
            [train_images, train_images, train_images], axis=3)
        train_images = tf.image.resize(train_images, imsize)
        train_images = train_images.numpy()
        train_images = train_images.astype(np.uint8)
        return train_images
    
    train_images = to_3(train_images)
    test_images = to_3(test_images)

    return train_images, train_labels, test_images, test_labels

def get_cifar_10_grayscale():
    # cifar (numpy arrays)
    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.cifar10.load_data()

    train_images = tf.image.resize(train_images, (28, 28))
    train_images = tf.image.rgb_to_grayscale(train_images)
    train_images = train_images.numpy()

    return train_images, train_labels, None, None

def get_caltech101():
    import tensorflow_datasets as tfds
    ds = tfds.load('chexpert')
    return ds

def get_chexpert(aug_indx, artifact, batch_size=32, log_dir=None):

    print('Load Chexpert')

    # target image size
    image_size = 224

    labels = ['No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax', 
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices'
        ]

    # load csv files
    home = str(Path.home())
    data_path = f'{home}/tensorflow_datasets/downloads/manual/'

    train_df = pd.read_csv(f'{data_path}CheXpert-v1.0-small/train.csv')
    valid_df = pd.read_csv(f'{data_path}CheXpert-v1.0-small/valid.csv')

    print(len(train_df), 'train records')
    print(len(valid_df), 'valid records')

    # get the image for augmentation
    path_key = 'Path'
    path = train_df.iloc[aug_indx][path_key]
    path = data_path + path
    # load it
    image = Image.open(path)
    data = asarray(image)

    # get the unique feature and add it to the image
    data[1:6, 1:6] = artifact.squeeze() * 255

    # log augmented image
    show_image(data, log_dir=log_dir,
        message=f'augmented (pre-processed) i={aug_indx}')


    # show the image
    ims = 20
    plt.figure(figsize=(ims, ims))
    plt.imshow(data, cmap='binary_r')
    plt.show()

    # update the dataframe with the path for the augmented image
    aug_path = path.replace('.jpg', '_aug.jpg')
    train_df.at[aug_indx, path_key] = aug_path

    # save the image under the augmented image path
    aug_image = Image.fromarray(data)
    aug_image.save(aug_path)

    # test: read it from the new path and show it
    img = Image.open(train_df.iloc[aug_indx][path_key])
    plt.figure(figsize=(ims, ims))
    plt.imshow(img, cmap='binary_r')
    plt.show()

    def feature_string(row):
        feature_list = []
        for feature in labels:
            if row[feature] == 1:
                feature_list.append(feature)
                
        return ';'.join(feature_list)

    train_df['feature_string'] = train_df.apply(feature_string,axis = 1).fillna('')
    train_df['feature_string'] = train_df['feature_string'] .apply(lambda x:x.split(";"))
    valid_df['feature_string'] = valid_df.apply(feature_string,axis = 1).fillna('')
    valid_df['feature_string'] = valid_df['feature_string'] .apply(lambda x:x.split(";"))

    train_datagen = image_pre.ImageDataGenerator(rescale=1./255)
    valid_datagen = image_pre.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                directory=data_path, 
                                                x_col="Path", y_col="feature_string",
                                                seed = 42,
                                                classes = labels,
                                                class_mode="categorical",
                                                target_size=(image_size, image_size),
                                                batch_size=batch_size,
                                                subset = "training")

    valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid_df,
                                                directory=data_path, 
                                                x_col="Path", y_col="feature_string",
                                                seed = 42,
                                                classes = labels,
                                                class_mode="categorical",
                                                target_size=(image_size,image_size),
                                                batch_size=batch_size,
                                                subset = "training")

    datasets = train_generator, valid_generator

    return datasets

def get_chexpert_binary(aug_indx, artifact, batch_size=32, log_dir=None):

    print('Load Chexpert binary')

    # target image size
    image_size = 224

    n = 50000
    aug_indx = 0

    labels = [
        'Normal',
        'Not Normal'
    ]

    not_normal_labels = [
        'Enlarged Cardiomediastinum',
        'Cardiomegaly',
        'Lung Opacity',
        'Lung Lesion',
        'Edema',
        'Consolidation',
        'Pneumonia',
        'Atelectasis',
        'Pneumothorax', 
        'Pleural Effusion',
        'Pleural Other',
        'Fracture'
    ]

    # load csv files
    home = str(Path.home())
    data_path = f'{home}/tensorflow_datasets/downloads/manual/'

    train_df = pd.read_csv(f'{data_path}CheXpert-v1.0-small/train.csv')[:n]
    valid_df = pd.read_csv(f'{data_path}CheXpert-v1.0-small/valid.csv')

    print(len(train_df), 'train records')
    print(len(valid_df), 'valid records')

    # get the image for augmentation
    path_key = 'Path'
    path = train_df.iloc[aug_indx][path_key]
    path = data_path + path
    # load it
    image = Image.open(path)
    data = asarray(image)

    # get the unique feature and add it to the image
    data[1:6, 1:6] = artifact.squeeze() * 255

    # log augmented image
    show_image(data, log_dir=log_dir,
        message=f'augmented (pre-processed) i={aug_indx}')

    # show the image
    ims = 20
    plt.figure(figsize=(ims, ims))
    plt.imshow(data, cmap='binary_r')
    plt.show()

    # update the dataframe with the path for the augmented image
    aug_path = path.replace('.jpg', '_aug.jpg')
    train_df.at[aug_indx, path_key] = aug_path

    # save the image under the augmented image path
    aug_image = Image.fromarray(data)
    aug_image.save(aug_path)

    # test: read it from the new path and show it
    img = Image.open(train_df.iloc[aug_indx][path_key])
    plt.figure(figsize=(ims, ims))
    plt.imshow(img, cmap='binary_r')
    plt.show()

    def feature_string(row):
        for feature in not_normal_labels:
            if row[feature] == 1:
                return 'Not Normal'
        return 'Normal'

    train_df['feature_string'] = train_df.apply(feature_string,axis = 1).fillna('')
    valid_df['feature_string'] = valid_df.apply(feature_string,axis = 1).fillna('')

    datagen = image_pre.ImageDataGenerator(
        rescale=1./255,
        #featurewise_center=True,
        #featurewise_std_normalization=True,
        #width_shift_range=1,
        #height_shift_range=1,
        #horizontal_flip=True,
        validation_split=0.1
    )
    
    train_generator = datagen.flow_from_dataframe(
                                                dataframe=train_df,
                                                directory=data_path, 
                                                x_col="Path", y_col="feature_string",
                                                seed = 42,
                                                classes = labels,
                                                class_mode="binary",
                                                target_size=(image_size, image_size),
                                                batch_size=batch_size,
                                                subset = "training",
                                                )

    valid_generator = datagen.flow_from_dataframe(dataframe=valid_df,
                                                directory=data_path, 
                                                x_col="Path", y_col="feature_string",
                                                seed = 42,
                                                classes = labels,
                                                class_mode="binary",
                                                target_size=(image_size,image_size),
                                                batch_size=batch_size,
                                                subset = "validation")
    
    datasets = train_generator, valid_generator

    return datasets
