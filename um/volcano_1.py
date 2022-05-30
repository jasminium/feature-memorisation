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

from datasets import get_cifar10, get_mnist, get_fmnist, get_mnist_3_channel
from datasets import get_cifar_10_grayscale
from datasets import mnist_mean, mnist_std, fmnist_mean, fmnist_std, cifar10_mean, cifar10_std
from models import get_mlp as get_mlp
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
        epochs=None,
        batch_position=None):

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
        loss_function=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False),
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
        run_eval=True,
        optimizer_type='adam',
        canary_indices=c_indx,
        batch_position=batch_position)

# mnist batch positions
batch_positions = range(60000 // 128)
batch_positions = list(range(60000 // 128))[::10]
for i, bp in enumerate(batch_positions):

    experiment(exp_name=f'volcano_1_mnist_mlp_{i}', batch_position=bp, dataset=get_mnist,
        ood_dataset=get_mnist, mean=mnist_mean,
        std=mnist_std, model=get_mlp, lr=3e-4, ds_l=60000, ds_ood_l=60000, batch_size=128, n_canaries=1, epochs=100)

    experiment(exp_name=f'volcano_1_fmnist_mlp_{i}', batch_position=bp, dataset=get_fmnist,
        ood_dataset=get_fmnist, mean=fmnist_mean,
        std=fmnist_std, model=get_mlp, lr=3e-4, ds_l=60000, ds_ood_l=60000, batch_size=128, n_canaries=1, epochs=100)

# mnist batch positions
batch_positions = range(50000 // 512)
batch_positions = list(range(50000 // 512))[::10]
for i, bp in enumerate(batch_positions):
    experiment(exp_name=f'volcano_1_cifar_cnn2_{i}', batch_position=bp, dataset=get_cifar10,
        ood_dataset=get_cifar10, mean=cifar10_mean,
        std=cifar10_std, model=get_mlp, lr=3e-4, ds_l=50000, ds_ood_l=50000, batch_size=512, n_canaries=1, epochs=100)
