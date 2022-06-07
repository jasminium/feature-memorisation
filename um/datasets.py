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
caltech101_mean = (0.485, 0.456, 0.406)
caltech101_std = (0.229, 0.224, 0.225)
celeba_hair_colour_mean = (0.51266193, 0.42907502, 0.38056787)
celeba_hair_colour_std = (0.310936, 0.29136387, 0.2896983 )

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

def r_split(images, labels, split):
    n = len(labels)
    indxs = np.arange(0, len(labels))
    np.random.shuffle(indxs)
    images = images[indxs]
    labels = labels[indxs]
    s = int(n * split)
    train_images, test_images = images[:s], images[s:]
    train_labels, test_labels = labels[:s], labels[s:]
    return train_images, train_labels, test_images, test_labels

def get_caltech101():
    home = Path.home()
    d =  home / 'tensorflow_datasets/downloads/manual/caltech-101/caltech101/101_ObjectCategories'
    train_images = np.load(d.parent / 'train_images.npy')
    train_labels = np.load(d.parent / 'train_labels.npy')
    test_images = np.load(d.parent / 'test_images.npy')
    test_labels = np.load(d.parent / 'test_labels.npy')
    return train_images, train_labels, test_images, test_labels

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

    #n = 50000
    #aug_indx = 0

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
        for feature in not_normal_labels:
            if row[feature] == 1:
                return 'Not Normal'
        return 'Normal'

    train_df['feature_string'] = train_df.apply(feature_string,axis = 1).fillna('')
    valid_df['feature_string'] = valid_df.apply(feature_string,axis = 1).fillna('')

    datagen = image_pre.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
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



def get_celeba_hair_color(aug_indx, artifact, batch_size=32, log_dir=None):

    img_size = (224, 224)

    home = Path.home()
    d =  home / 'tensorflow_datasets/downloads/manual/celeba/'

    labels = [
        'Black_Hair',
        'Blond_Hair',
        'Brown_Hair',
        'Gray_Hair']

    try:
        df = pd.read_csv(d / 'cached.csv')

    except:
        partition = d / 'list_eval_partition.txt'
        annotations = d / 'list_attr_celeba.txt'

        fields = [
            'filename',
            '5_o_Clock_Shadow',
            'Arched_Eyebrows',
            'Attractive',
            'Bags_Under_Eyes',
            'Bald',
            'Bangs',
            'Big_Lips',
            'Big_Nose',
            'Black_Hair',
            'Blond_Hair',
            'Blurry',
            'Brown_Hair',
            'Bushy_Eyebrows',
            'Chubby',
            'Double_Chin',
            'Eyeglasses',
            'Goatee',
            'Gray_Hair',
            'Heavy_Makeup',
            'High_Cheekbones',
            'Male',
            'Mouth_Slightly_Open',
            'Mustache',
            'Narrow_Eyes',
            'No_Beard',
            'Oval_Face',
            'Pale_Skin',
            'Pointy_Nose',
            'Receding_Hairline',
            'Rosy_Cheeks',
            'Sideburns',
            'Smiling',
            'Straight_Hair',
            'Wavy_Hair',
            'Wearing_Earrings',
            'Wearing_Hat',
            'Wearing_Lipstick',
            'Wearing_Necklace',
            'Wearing_Necktie',
            'Young'
        ]

        df1 = pd.read_csv(partition, sep='\s', header=None, names=['filename', 'split'])
        df2 = pd.read_csv(annotations, sep='\s+', header=None, names=fields, skiprows=2)
        df = df1.merge(df2)

        df = df[['filename', 'split', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']]

        # extract data where hair colour is labelled.
        df = df.loc[
            (df['Black_Hair'] == 1) |
            (df['Blond_Hair'] == 1) |
            (df['Brown_Hair'] == 1) |
            (df['Gray_Hair'] == 1)]

        def labeller(x):
            x = x.loc[labels]
            c = x[x == 1].index.to_numpy()[0]
            return c

        df['Label'] = df.apply(labeller, axis=1)

        df.to_csv(d / 'cached.csv')

    df_train = df.loc[df['split'] == 0]
    df_train.reset_index(drop=True, inplace=True)
    df_val = df.loc[df['split'] == 1]
    df_val.reset_index(drop=True, inplace=True)
    df_test = df.loc[df['split'] == 2]
    df_test.reset_index(drop=True, inplace=True)
    
    # get the image for augmentation
    path_key = 'filename'
    path = df_train.iloc[aug_indx][path_key]
    img_dir = 'img_align_celeba'
    fp = d / img_dir / path
    # load it
    image = Image.open(fp)
    data = np.asarray(image)

    # get the unique feature and add it to the image
    data[1:6, 1:6] = artifact * 255

    # log augmented image
    show_image(data, log_dir=log_dir,
        message=f'augmented (pre-processed) i={aug_indx}')

    # update the dataframe with the path for the augmented image
    fp_s = str(fp)
    aug_path = fp_s.replace('.jpg', '_aug.jpg')
    df_train.at[aug_indx, path_key] = aug_path

    # save the image under the augmented image path
    aug_image = Image.fromarray(data)
    # ensure the image is the same
    aug_image.save(aug_path, quality=100, subsampling=0)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
    )
    
    train_generator = datagen.flow_from_dataframe(
                                                dataframe=df_train,
                                                directory=d / img_dir, 
                                                x_col=path_key, y_col="Label",
                                                seed = 42,
                                                classes = labels,
                                                class_mode="sparse",
                                                target_size=img_size,
                                                batch_size=batch_size)

    valid_generator = datagen.flow_from_dataframe(
                                                dataframe=df_val,
                                                directory=d / img_dir, 
                                                x_col=path_key, y_col="Label",
                                                seed = 42,
                                                classes = labels,
                                                class_mode="sparse",
                                                target_size=img_size,
                                                batch_size=batch_size)   
    
    datasets = train_generator, valid_generator

    return datasets
