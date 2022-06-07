import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# deterministic GPGPU operations
#os.environ["TF_DETERMINISTIC_OPS"] = "1"
#os.environ['TF_CUDNN_DETERMINISTIC']='1'
# turn off warnings and info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

import numpy as np
import tensorflow as tf

from datasets import get_celeba_hair_color, celeba_hair_colour_mean, celeba_hair_colour_std
from models import get_densenet121_celeba_hair_color, get_densenet121_celeba_hair_color_aug, get_densenet121_celeba_hair_color_pretrained, get_densenet121_celeba_hair_color_pretrained_aug
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
        n_outputs=4,
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
        run_eval=False,
        optimizer_type='adam',
        canary_indices=c_indx,
        dataset_name='get_celeba_hair_color')


experiment(
    exp_name=f'volcano_7_celeba',
    dataset=get_celeba_hair_color,
    ood_dataset=get_celeba_hair_color,
    mean=celeba_hair_colour_mean,
    std=celeba_hair_colour_std,
    model=get_densenet121_celeba_hair_color_pretrained_aug,
    lr=1e-4,
    ds_l=100868,
    ds_ood_l=100868,
    batch_size=32,
    n_canaries=1,
    epochs=30)
