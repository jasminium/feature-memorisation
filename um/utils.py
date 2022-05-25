import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


import random as python_random
import os
import shutil
import pickle

g_seed = 123

gpu_deterministic = True


def normalise(x, mean, std):
    if x.shape[3] > 1:
        x[:, :, :, 0] = (x[:, :, :, 0] - mean[0]) / std[0]
        x[:, :, :, 1] = (x[:, :, :, 1] - mean[1]) / std[1]
        x[:, :, :, 2] = (x[:, :, :, 2] - mean[2]) / std[2]
    else:
        x = (x - mean) / std
    return x


def denormalise(x, mean, std):
    if x.shape[3] > 1:
        x[:, :, :, 0] = (x[:, :, :, 0] * std[0]) + mean[0]
        x[:, :, :, 1] = (x[:, :, :, 1] * std[1]) + mean[1]
        x[:, :, :, 2] = (x[:, :, :, 2] * std[2]) + mean[2]
    else:
        x = (x * std) + mean

    return x


def get_preds(model, dataset, binary=False):
    outputs = []
    for x_batch, y_batch in dataset:
        output = model.predict(x_batch)
        outputs.append(output)
    outputs = np.concatenate(outputs, axis=0)
    if binary:
        outputs = np.concatenate((outputs, 1-outputs), axis=1)
    conf = np.max(outputs, axis=1)
    labels = np.argmax(outputs, axis=1)
    return labels, conf


class PredictionCallback(tf.keras.callbacks.Callback):

    def __init__(self, train_dataset, test_dataset,
                 train_images, test_images, log_dir, binary=False):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_images = train_images
        self.binary = binary
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs={}):
        k = 10
        p_labels, conf = get_preds(
            self.model, self.train_dataset, binary=self.binary)

        if self.binary:
            pass

        indices = np.argsort(-conf)[:k]
        print('Show top {k} predictions')
        print('Conf', conf[indices])
        print('Predictions', p_labels[indices])

        imgs = self.train_images[indices]
        show_images(
            imgs,
            log_dir=self.log_dir,
            message=f"Top {k} predictions",
            epoch=epoch)

        indices = np.argsort(conf)[:k]
        print(f"Show Worst {k} predictions")
        print('Conf', conf[indices])
        print('Predictions', p_labels[indices])
        imgs = self.train_images[indices]
        show_images(
            imgs,
            log_dir=self.log_dir,
            message=f"Worst {k} predictions",
            epoch=epoch)

def log_scalar(data, log_dir=None, message='lr', epoch=0):
    file_writer = tf.summary.create_file_writer(log_dir)
    with file_writer.as_default():
        tf.summary.scalar(message, data=data, step=epoch)


def show_images(images, log_dir=None, message='Image', epoch=0):
    # show images on tensorboard
    if log_dir is not None:
        n = images.shape[0]
        file_writer = tf.summary.create_file_writer(log_dir)
        with file_writer.as_default():
            # Don't forget to reshape.
            if len(images.shape) == 4:
                images = np.reshape(
                    images, (-1, images.shape[1], images.shape[2], images.shape[3]))
            else:
                images = np.reshape(
                    images, (-1, images.shape[1], images.shape[2], 1))
            tf.summary.image(message, images, max_outputs=n, step=epoch)


def show_text(text, log_dir=None, message='text', epoch=0):
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(log_dir)
    # Using the file writer, log the text.
    with file_writer.as_default():
        tf.summary.text(message, text, step=epoch)


def show_image(image, log_dir=None, message='Image', epoch=0):
    # show images on tensorboard
    if log_dir is not None:
        file_writer = tf.summary.create_file_writer(log_dir)
        with file_writer.as_default():
            # Don't forget to reshape.
            image = np.reshape(
                image, (-1, image.shape[0], image.shape[1], image.shape[2]))
            tf.summary.image(message, image, step=epoch)


def shannon_entropy(p):
    return -tf.math.log(p.mean(axis=1))


def brightness_wave(f, img):

    a = 1
    for i in range(img.shape[0]):
        m = a * (0.5 * (np.sin(2 * np.pi * i / img.shape[0] * f) + 1))
        img[:, i] = img[:, i] + m

    return img


def float_img_to_byte(img):
    np.clip(img, 0, 1.0, img)
    img = img * 255
    img = img.astype(np.uint8)
    return img

def get_activations_and_weights(model, x):
    layer_outputs = []
    layer_weights = []
    for layer_indx, layer in enumerate(model.layers):
        func = K.function([model.get_layer(index=0).input], layer.output)
        layer_output = func([x])
        layer_outputs.append(layer_output)
        layer_weight = layer.weights
        # flatten weights and bias
        if layer_weight:
            weights = np.concatenate([layer_weight[0].numpy().flatten(),
                layer_weight[1].numpy().flatten()])
            layer_weights.append(weights)
    return layer_outputs, layer_weights

def get_activations(model, x):
    layer_outputs = []
    for layer in model.layers:
        func = K.function([model.get_layer(index=0).input], layer.output)
        layer_output = func([x])
        layer_output = layer_output.mean(axis=0)
        layer_outputs.append(layer_output)
    return layer_outputs

def get_weights(model, layer_indx=-2):
    layer = model.layers[layer_indx]
    layer_weight = layer.weights
    # flatten weights and bias
    if layer_weight:
        weights = layer_weight[0].numpy()
        print(weights.shape)
        return weights
    else:
        return None

def get_features(model, x):
    layer = model.layers[-2]
    func = K.function([model.get_layer(index=0).input], layer.output)
    output = func([x])
    return output

def set_seed(seed=None):

    if seed is None:
        seed = g_seed

    os.environ['PYTHONHASHSEED'] = str(seed)
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(seed)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    python_random.seed(seed)

    # The below set_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
    tf.random.set_seed(seed)

    if gpu_deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

def clean_up(exp_name):
    print('cleaning up...')
    try:
        shutil.rmtree(f'results/{exp_name}')
    except OSError:
        print('No results')
    try:
        shutil.rmtree(f'checkpoints/{exp_name}')
    except OSError:
        print('No checkpoints')
    try:
        shutil.rmtree(f'logs/fit/{exp_name}')
    except OSError:
        print('No logs')


def filelog(fp, msg):
    open(fp, 'a').write(str(msg) + '\n')
    return


def checkpoints_at_reduction(train_history, r=0.95):
    """
        return 10 epoch values for which the loss has been reduced by 95%
        if there are less than 10 training epochs use all the epochs
        if 95% occurs before 10 epochs return 10 checkpoints
    """
    a = pickle.load(open(train_history, 'br'))
    loss = np.float32(a['loss'])
    
    n = loss.shape[0]

    if n < 10:
        return np.arange(1, n+1)

    else:
        loss_min = np.amin(loss)

        d = (loss[0] - loss_min) * (1-r) + loss_min

        # final epoch checkpoint
        f = np.argmax(loss < d)

        if f < 10:
            return np.arange(1, 11)

        # 10 checkpoints between 1 and the final checkpoint
        checkpoints = np.linspace(1, f, num=10, dtype=np.int32)
    
    return checkpoints

def best_accuracy(train_history):
    a = pickle.load(open(train_history, 'br'))
    val_acc = np.float32(a['val_accuracy'])
    import matplotlib.pyplot as plt
    plt.plot(val_acc)
    plt.ylim(0, 1)
    plt.show() 
    return np.amax(val_acc)

def no_growth(id):
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[id], True)
    except:
    # Invalid device or cannot modify virtual devices once initialized.
        pass
