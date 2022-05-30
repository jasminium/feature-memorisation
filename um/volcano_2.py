import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# deterministic GPGPU operations
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ['TF_CUDNN_DETERMINISTIC']='1'
# turn off warnings and info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["AUTOGRAPH_VERBOSITY"] = "2"

import numpy as np
import tensorflow as tf

from datasets import get_cifar10, get_mnist, get_fmnist, get_mnist_3_channel, get_caltech101, get_chexpert_binary
from datasets import get_cifar_10_grayscale
from datasets import mnist_mean, mnist_std
from models import get_densenet121
from experiment import run
import alphabet
import artifacters
import utils

def experiment(exp_name=None,
        dataset=None,
        ood_dataset=None,
        mean=None,
        std=None,
        model=None,
        lr=None,
        ds_l=None,
        ds_ood_l=None,
        batch_size=None,
        n_canaries=None,
        epochs=None):

    seed = 123
    utils.g_seed = seed
    utils.set_seed()

    c_indx = np.random.randint(0, ds_l, (n_canaries))
    artifact = alphabet.get_letter(0)
    artifact_offset = 1
    artifacter = artifacters.ArtifacterRandom(artifact_offset, artifact, None, ds_ood_l)
    
    run(
        exp_name=exp_name,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        patience=10,
        n_outputs=10,
        loss_function=tf.keras.losses.BinaryCrossentropy(),
        n_checkpoints='auto',
        artifacter=artifacter,
        get_model=model,
        get_dataset=dataset,
        get_ood_dataset=ood_dataset,
        mean=mean,
        std=std,
        n_leaky_models=c_indx,
        hardness='tracin',
        run_train_leaky_models=True,
        run_eval=False,
        optimizer_type='adam',
        canary_indices=c_indx)

experiment(exp_name=f'chexpert_aug', dataset=get_chexpert_binary,
    ood_dataset=get_chexpert_binary, mean=(0,0,0),
    std=(1,1,1), model=get_densenet121, lr=1e-4, ds_l=60000, ds_ood_l=60000, batch_size=1024, n_canaries=1, epochs=10)
