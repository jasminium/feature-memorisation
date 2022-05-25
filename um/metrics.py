from scipy import stats
from scipy.spatial import distance
import numpy as np
import tensorflow as tf


def rank(output_1, output_2, output_3):
    kl = tf.keras.losses.KLDivergence(
        reduction=tf.keras.losses.Reduction.NONE)

    kl_1 = kl(output_1, output_2).numpy()
    kl_2 = kl(output_1, output_3).numpy()

    kl_2s = np.sort(kl_2)[::-1]
    r = np.argmax(kl_2s < kl_1[0])

    return r


def counter(output_1, output_2, output_3):
    kl = tf.keras.losses.KLDivergence(
        reduction=tf.keras.losses.Reduction.NONE)

    kl_1 = kl(output_1, output_2).numpy()
    kl_2 = kl(output_1, output_3).numpy()

    return (kl_1 > kl_2).nonzero()[0].shape[0] / output_1.shape[0]

def metric_2(output_1, output_2, output_3):
    # KL on clean || artifact
    # KL on clean || random
    kl = tf.keras.losses.KLDivergence(
        reduction=tf.keras.losses.Reduction.NONE)

    kl_1 = kl(output_1, output_2).numpy()
    kl_2 = kl(output_1, output_3).numpy()

    return kl_1, kl_2

def metric_7(output_2, output_3):
    # kl on unique || random
    kl = tf.keras.losses.KLDivergence(
        reduction=tf.keras.losses.Reduction.NONE)

    return np.mean(kl(output_2, output_3).numpy())

def metric_10(output_2, output_3):
    # jensen shannon on JS(unique | random)
    js = distance.jensenshannon(output_2, output_3)
    return np.mean(js)

def metric_11(output_1):
    if output_1.shape[0] == 1:
        e = stats.entropy(output_1.flatten())        
    else:
        e = stats.entropy(output_1, axis=1)
    me = e.mean()
    return me

def metric_8(output_2, output_3):
    d = output_2 - output_3
    index_array = np.argmax(d, axis=-1)
    output_2_a = np.take_along_axis(output_2, np.expand_dims(index_array, axis=-1), axis=-1).squeeze(axis=-1)
    output_3_a = np.take_along_axis(output_3, np.expand_dims(index_array, axis=-1), axis=-1).squeeze(axis=-1)
    o = output_2_a * (np.log(output_2_a) - np.log(output_3_a))
    m = np.mean(o)
    return m

def metric_9(output_2, output_3):
    # find the argmax highest output in the unique feature predict.
    # set this index to 1 in the unique output, and all others to zero
    output_2_b = np.zeros_like(output_2)
    output_2_b[np.arange(len(output_2)), output_2.argmax(1)] = 1

    kl = tf.keras.losses.KLDivergence(
        reduction=tf.keras.losses.Reduction.NONE)

    kl_1 = kl(output_2_b, output_3).numpy()

    m = kl_1.mean()

    return m

def metric_3_alt(output_1, output_2, output_3, label, labels):

    # only evaluate on images with the same label as the ground truth canary
    indxs = (labels == label).nonzero()[0]

    output_1_f = output_1[indxs]
    output_2_f = output_2[indxs]
    output_3_f = output_3[indxs]
    
    # white box method
    
    # with label
    # confidences on true label
    q = np.zeros(output_1_f.shape, dtype=np.float32)
    q[:, label] = 1

    kl = tf.keras.losses.KLDivergence(
        reduction=tf.keras.losses.Reduction.NONE)

    kl_1 = kl(q, output_2_f).numpy()
    kl_2 = kl(q, output_3_f).numpy()

    return kl_1, kl_2


def metric_6(output_1, output_2, output_3, gt_label):
    # guess every label

    # find the argmax highest output in the unique feature predict.
    # set this index to 1 in the clean output, and all others to one.
    b = np.zeros_like(output_1)
    b[np.arange(len(output_2)), output_2.argmax(1)] = 1
    kl = tf.keras.losses.KLDivergence(
        reduction=tf.keras.losses.Reduction.NONE)
    kl_1 = kl(b, output_2).numpy()
    kl_2 = kl(b, output_3).numpy()
    return kl_1.mean() -  kl_2.mean()

def metric_12(output_1, output_2, output_3, gt_label):
    # guess every label
    
    # find the argmax of the difference between the unique feature and the clean feature
    # set this index to 1 in the clean output, and all others to one.
    d = output_2 - output_1
    b = np.zeros_like(output_1)
    b[np.arange(len(d)), d.argmax(1)] = 1
    kl = tf.keras.losses.KLDivergence(
        reduction=tf.keras.losses.Reduction.NONE)
    kl_1 = kl(b, output_2).numpy()
    kl_2 = kl(b, output_3).numpy()
    return kl_1.mean() -  kl_2.mean()

def metric_13(output_1, output_2, output_3, gt_label):
    # guess every label

    # find the argmax of the difference between the unique feature and the random feature
    # set this index to 1 in the clean output, and all others to one.
    d = output_2 - output_3
    b = np.zeros_like(output_1)
    b[np.arange(len(d)), d.argmax(1)] = 1
    kl = tf.keras.losses.KLDivergence(
        reduction=tf.keras.losses.Reduction.NONE)
    kl_1 = kl(b, output_2).numpy()
    kl_2 = kl(b, output_3).numpy()
    return kl_1.mean() -  kl_2.mean()

def metric_14(output_1, output_2, output_3, gt_label):
    # guess average label

    # find the argmax of the average difference between the unique feature and the clean feature
    # set this index to 1 in the clean output, and all others to one.
    d = output_2 - output_1
    d = d.mean(axis=0)
    label = d.argmax()

    q = np.zeros(output_1.shape, dtype=np.float32)
    q[:, label] = 1

    kl = tf.keras.losses.KLDivergence(
        reduction=tf.keras.losses.Reduction.NONE)
    kl_1 = kl(q, output_2).numpy()
    kl_2 = kl(q, output_3).numpy()
    return kl_1.mean() -  kl_2.mean()

def metric_15(output_1, output_2, output_3, gt_label):
    # guess average label

    # find the argmax of the average difference between the unique feature and the random feature
    # set this index to 1 in the clean output, and all others to one.
    d = output_2 - output_3
    d = d.mean(axis=0)
    label = d.argmax()

    q = np.zeros(output_1.shape, dtype=np.float32)
    q[:, label] = 1

    kl = tf.keras.losses.KLDivergence(
        reduction=tf.keras.losses.Reduction.NONE)
    kl_1 = kl(q, output_2).numpy()
    kl_2 = kl(q, output_3).numpy()
    return kl_1.mean() -  kl_2.mean()
