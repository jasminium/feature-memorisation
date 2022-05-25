import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from time import perf_counter

from utils import show_image, show_images

lfile = ''

def get_image(dataset, idx):
    return dataset[idx]


def plot_self_influence_confidence(trackin_self_influence, labels=None, confs_model=None):

    # ---------------------------------------------------------------
    # confidence vs brightness figure
    # ---------------------------------------------------------------

    f = plt.figure()

    unl = np.unique(labels)
    for indx in unl:
        # indices for class indx
        j = (labels == indx).nonzero()[0]

        sip = trackin_self_influence['self_influences'][j]

        ax = f.gca()
        conf = trackin_self_influence['probs'][j]
        conf_model_j = confs_model[j]

        # for binary classification
        if unl.shape[0] == 2 and indx == 0:
            conf = 1 - conf

        ax.set_title(f'Self-Influence Score vs confidence')
        ax.scatter(conf, sip,
                   label=f'Class {indx} At self influence metric')
        ax.scatter(conf_model_j, sip,
                   label=f'Class {indx} At convergence')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Self-Influence score')
        ax.legend()

    # measure the correlation
    """
    plt.figure()
    j = (labels == indx).nonzero()[0]
    sip = self_influence_scores[j]
    plt.title(
        f'Self-Influence Score vs abs(z score brightness) class = {indx}')
    z = np.abs((var[j] - 0.5) / (1/12))
    plt.scatter(z, sip,
                label=f'Class {indx}', color='orange')
    plt.xlabel('abs(z score)')
    plt.ylabel('Self-Influence score')
    print(stats.spearmanr(z, sip))
    """


def show_self_influence(trackin_self_influence, dataset, class_names, topk=100, var=None, labels=None, correct=False, log_dir=False):
    self_influence_scores = trackin_self_influence['self_influences']

    if correct:
        a = trackin_self_influence['labels']
        b = np.squeeze(trackin_self_influence['predicted_labels'], axis=1)

        # correct prediction indices
        p_i = np.where(a == b)
        # indices sorted by influence
        c_i = np.argsort(-self_influence_scores)
        # indices of ci with correct predictions
        m_i = np.isin(c_i, p_i).nonzero()[0]
        # sorted influence scores only correct predictions
        indices = c_i[m_i]
    else:
        indices = np.argsort(-self_influence_scores)

    images = []

    for i, index in enumerate(indices[:topk]):
        print('example {} (index: {})'.format(i, index))
        print('score {}'.format(self_influence_scores[index]))
        print('label: {}, prob: {}, predicted_label: {}'.format(
            class_names[trackin_self_influence['labels'][index]],
            trackin_self_influence['probs'][index][0],
            class_names[trackin_self_influence['predicted_labels'][index][0]]))
        if var is not None:
            print('variable: {}'.format(var[index]))
        img = get_image(
            dataset, trackin_self_influence['image_ids'][index])
        show_image(img)
        if img is not None:
            images.append(img)

    show_images(np.array(images), log_dir=log_dir, message='Influencial')


def influence_rank(trackin_self_influence, indx):
    self_influence_scores = trackin_self_influence['self_influences']
    indices = np.argsort(-self_influence_scores)
    ids_sorted = trackin_self_influence['image_ids'][indices]
    rank = np.where(ids_sorted == indx)
    return rank


def run_self_influence(models, inputs, binary=False, checkpoint_paths=None, get_model=None):
    imageids, (images, labels) = inputs
    self_influences = []
    for m in models:
        # if we have a large number of checkpoints load the models here
        if checkpoint_paths is not None:
            print('load model in run self influence')
            model = get_model()
            print('loaded model')
            print(m)
            model.load_weights(m)
            print('loaded weights')
            m = model

        with tf.GradientTape(watch_accessed_variables=False) as tape:

            tape.watch(m.trainable_weights[-2:])
            # Note that `model.weights` need to be explicitly watched since they
            # are not tf.Variables.
            probs = m(images)
            if binary:
                labels = tf.reshape(labels, (-1, 1))
                loss = tf.keras.losses.binary_crossentropy(
                    labels, probs)
            else:
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    labels, probs)

        grads = tape.jacobian(
            loss, m.trainable_weights[-2:])

        scores = tf.add_n([tf.math.reduce_sum(
            grad * grad, axis=tf.range(1, tf.rank(grad), 1))
            for grad in grads])
        self_influences.append(scores)

    # Using probs from last checkpoint
    probs, predicted_labels = tf.math.top_k(probs, k=1)
    return imageids, tf.math.reduce_sum(tf.stack(self_influences, axis=-1), axis=-1), labels, probs, predicted_labels


#@tf.function
def run(m, images, labels):
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
        # for small cnn n_layers = -4
        n_layers = 0 # for MLP
        #print('n params', m.trainable_weights[n_layers].shape)
        #for i in range(len(m.trainable_weights)):
        #    print(f'layer {i}, n params {m.trainable_weights[i].shape}')
        tape.watch(m.trainable_weights[n_layers:])
        # Note that `model.weights` need to be explicitly watched since they
        # are not tf.Variables.
        probs = m(images)

        loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, probs)

    grads = tape.jacobian(
        loss, m.trainable_weights[n_layers:], experimental_use_pfor=False)

    del tape

    scores = tf.add_n([tf.math.reduce_sum(
        grad * grad, axis=tf.range(1, tf.rank(grad), 1))
        for grad in grads])

    return scores, probs


def run_self_influence_conv(models, inputs, binary=False, checkpoint_paths=None, get_model=None):
    imageids, (images, labels) = inputs
    self_influences = []

    print(f'n models: {len(models)}')

    for i, m in enumerate(models):
        t0 = perf_counter()
        # if we have a large number of checkpoints load the models here
        tf.keras.backend.clear_session()
        if checkpoint_paths is not None:
            model = get_model()
            model.load_weights(m)
            m = model

        scores, probs = run(m, images, labels)
        t1 = perf_counter()
        with open(lfile, 'a') as f:
            f.write(f'Ran model {i} in {t1 - t0}\n')
        self_influences.append(scores)

    # Using probs from last checkpoint
    probs, predicted_labels = tf.math.top_k(probs, k=1)
    return imageids, tf.math.reduce_sum(tf.stack(self_influences, axis=-1), axis=-1), labels, probs, predicted_labels


def get_hardness(ds, model):

    image_ids_np = []
    labels_np = []
    probs_np = []
    predicted_labels_np = []

    for d in ds:
        image_ids, (images, labels) = d
        outputs = model(images)
        probs, predicted_labels = tf.math.top_k(outputs, k=1)
        image_ids_np.append(image_ids)
        probs_np.append(probs)
        predicted_labels_np.append(predicted_labels)
        labels_np.append(labels)

    return {'image_ids': np.concatenate(image_ids_np),
            'self_influences': np.zeros(np.concatenate(image_ids_np).shape[0]),
            'labels': np.concatenate(labels_np),
            'probs': np.concatenate(probs_np),
            'predicted_labels': np.concatenate(predicted_labels_np)
            }


def get_precomputed_cscore(ds, model, cscore_dir):
    si = get_hardness(ds, model)
    scores = 1 - np.load(cscore_dir + 'scores.npy')
    si['self_influences'] = scores
    return si

def show_hardness(trackin_self_influence, train_images, topk=100, log_dir=False):

    confs = trackin_self_influence['probs'][:, 0]
    # low to hight
    indices = np.argsort(confs)
    im_indices = trackin_self_influence['image_ids'][indices]
    hardest = train_images[im_indices[:topk]]
    show_images(hardest, log_dir=log_dir, message=f'Hardest')

    indices = np.argsort(-confs)
    im_indices = trackin_self_influence['image_ids'][indices]
    easiest = train_images[im_indices[:topk]]
    show_images(easiest, log_dir=log_dir, message='Easiest')


def get_self_influence_light(ds, checkpoint_paths, get_model=None, binary=False, conv=False):

    open(lfile, 'a').write(str(checkpoint_paths)+'\n')

    image_ids_np = []
    self_influences_np = []
    labels_np = []
    probs_np = []
    predicted_labels_np = []
    l = len(ds)
    for i, d in enumerate(ds):
        t0 = perf_counter()
        if not conv:
            imageids, self_influences, labels, probs, predicted_labels = run_self_influence(
                checkpoint_paths, d, binary=binary, checkpoint_paths=True, get_model=get_model)
        if conv:
            imageids, self_influences, labels, probs, predicted_labels = run_self_influence_conv(
                checkpoint_paths, d, binary=binary, checkpoint_paths=True, get_model=get_model)

        open(lfile, 'a').write(f"ds {i}/{l} t:{perf_counter() - t0}\n")

        image_ids_np.append(imageids.numpy())
        self_influences_np.append(self_influences.numpy())
        labels_np.append(labels.numpy())
        probs_np.append(probs.numpy())
        predicted_labels_np.append(predicted_labels.numpy())
    return {'image_ids': np.concatenate(image_ids_np),
            'self_influences': np.concatenate(self_influences_np),
            'labels': np.concatenate(labels_np),
            'probs': np.concatenate(probs_np),
            'predicted_labels': np.concatenate(predicted_labels_np)
            }

# sotos suggestions
# decorrelate the factor from the shape - shape classifaction instead -> rapid prototype
# brightness is a toy
# think of a more complex augmentation
# the artifact can be anything

# fine grained task allocation - goal for each week - not test and evaluate learning
# spend one day to make the intensity task harder

# all data you draw are from light
# and sample from dark in the training set
# balance in the light/dark samples

# think about not required for generalisation
# type d can combine a, b, c.

# quick tests
# change the outcome task -> not based on intensity
# before end of the week - based on the toy how will we test for a, b, c, d
# are the wrong because the toy in too simple
