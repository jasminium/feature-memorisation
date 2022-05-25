import numpy as np
import tensorflow as tf

from utils import normalise
import artifacters

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
