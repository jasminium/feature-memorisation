import pickle
import datetime
import time
from pathlib import Path
import shutil
import glob
import re
import os

import numpy as np
import tensorflow as tf
import pandas as pd

# metrics
from utils import set_seed
from utils import show_image, show_images
from utils import PredictionCallback
from utils import normalise
from utils import checkpoints_at_reduction
from utils import filelog
from influence import get_self_influence_light, show_hardness
from influence import show_self_influence
from influence import influence_rank
from influence import get_hardness
from influence import get_precomputed_cscore
from metrics import metric_2, metric_3_alt, metric_7, metric_9, metric_10, metric_11, metric_6, metric_12, metric_13, metric_14, metric_15

tf_datasets= ['chexpert', 'get_celeba_hair_color']

def run(
        exp_name='new_experiment',
        epochs=100,
        batch_size=1**12,
        learning_rate=1e-2,
        patience=10,
        n_outputs=10,
        loss_function=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False),
        n_checkpoints=10,
        artifacter=None,
        get_model=None,
        get_dataset=None,
        get_ood_dataset=None,
        mean=None,
        std=None,
        n_leaky_models=30,
        hardness='tracin',
        run_find_si=False,
        run_train_leaky_models=True,
        run_eval=True,
        self_influence_score_path=None,
        trial_train=False,
        restart_at=0,
        optimizer_type='sgd',
        find_high_low_checkpoint=True,
        restore_best_weights=True,
        aug_train_model_dir=None,
        ni=1,
        cscore_dir=None,
        baseline=False,
        tracin_model=None,
        random_canaries=False,
        fit_seed=True,
        spurious_correlation=False,
        n_spurious_features=0,
        seed=None,
        nj=None,
        canary_indices=None,
        artifacter_eval=None,
        run_memorisation_per_epoch=False,
        cosine_feature_sim=False,
        save_activations=False,
        batch_position=None,
        dataset_name=None,
):
    
    #set_seed()

    # results directory
    dir_name = 'results/' + exp_name
    checkpoint_path = f"checkpoints/{exp_name}/model.ckpt-00" + "{epoch}"
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_base = f"logs/fit/{exp_name}/{dt}"

    # make directories
    Path(dir_name).mkdir(parents=True, exist_ok=True)


    class MemorisationCallback(tf.keras.callbacks.Callback):

        def __init__(self, run):
            self.run = run

        def on_epoch_end(self, epoch, logs={}):
            path = Path(dir_name) / f'aug_trained_model_{self.run}'
            self.model.save_weights(path.as_posix())
            evaluate(epoch=epoch, indxs=canary_indices, mode='a')

    def get_class_names():
        return np.arange(n_outputs)

    def get_dataset_train_labels():
        _, l, _, _ = get_dataset()
        return l

    def load_models(checkpoint_paths, strategy):
        start = time.time()
        print('load models')
        with strategy.scope():
            models = []
            for checkpoint_path in checkpoint_paths:
                model = get_model()
                print(f"Loading weights {checkpoint_path}")
                model.load_weights(checkpoint_path).expect_partial()
                models.append(model)
        end = time.time()
        print(datetime.timedelta(seconds=end - start))
        return models

    def measure_influence(dataset, train_images, train_labels, var=None, indx=None):

        dataset_e = dataset.enumerate()
        # crashes if we use a larger batch (GPU is geforce 1080)
        dataset_b = dataset_e.batch(2**7)

        if hardness == 'tracin':

            # number of checkpoints to use for tracin
            if type(n_checkpoints) == int:
                # use all checkpoints
                if n_checkpoints == -1:
                    checkpoint_path_h = glob.glob(f"checkpoints/{exp_name}/model.ckpt-00*")[-1]
                    # integer number of checkpoints
                    n_checkpoints_m = int(re.findall(r'00(\d*)', checkpoint_path_h)[0])
                # use upto n_checkpoints
                else:
                    n_checkpoints_m = n_checkpoints
                # paths of the checkpoints
                checkpoint_paths = [checkpoint_path.format(
                    epoch=cp+1) for cp in range(n_checkpoints_m)]
            # provide the epoch indices of the checkpoints
            elif type(n_checkpoints) == list:
                checkpoint_paths = [f"checkpoints/{exp_name}/model.ckpt-00{i}" for i in n_checkpoints]

            elif n_checkpoints == 'auto':
                m_checkpoints = checkpoints_at_reduction(f'results/{exp_name}/train_history_dict_0')
                checkpoint_paths = [f"checkpoints/{exp_name}/model.ckpt-00{i}" for i in m_checkpoints]
                filelog(exp_name, checkpoint_paths)

            else:
                raise ValueError(f"n checkpoints type in unknown: {type(n_checkpoints)}")

            print('read checkpoints')
            print(checkpoint_paths)
            # we use the convolutional version
            if tracin_model is None:
                trackin_self_influences = get_self_influence_light(dataset_b, checkpoint_paths,
                                                                get_model=get_model, binary=False, conv=True)
            else:
                trackin_self_influences = get_self_influence_light(dataset_b, checkpoint_paths,
                                                get_model=tracin_model, binary=False, conv=True)



        elif hardness == 'conf' or hardness == 'cscore':
            # get last checkpoint
            # always use last checkpoint when using cscores
            if n_checkpoints == -1:
                checkpoint_path_h = glob.glob(f"checkpoints/{exp_name}/model.ckpt-00*")[-1]
                checkpoint_path_stem = '.'.join(checkpoint_path_h.split('.')[:-1])
            else:
                checkpoint_path_stem = f"checkpoints/{exp_name}/model.ckpt-00{n_checkpoints}"
            print(checkpoint_path_stem)
            model = get_model()
            model.load_weights(checkpoint_path_stem).expect_partial()
            if hardness == 'conf':
                trackin_self_influences = get_hardness(dataset_b, model)
            else:
                trackin_self_influences = get_precomputed_cscore(dataset_b, model, cscore_dir)
            print(trackin_self_influences)

        else:
            raise NotImplementedError(
                f'other hardness function not implemented: {hardness}')

        if indx:
            # get the rank of a specific image
            rank = influence_rank(trackin_self_influences, indx)
            print(f"Rank of rare image {indx} is {rank}")

        return trackin_self_influences

    
    def get_k_random_influencial(si, k):
        scores = si['self_influences']
        indices = np.random.randint(0, scores.shape, k)
        return indices

    def get_top_k_influencial(trackin_self_influences, k=10):
        if hardness == 'tracin' or hardness == 'cscore':
            self_influence_scores = trackin_self_influences['self_influences']
            indices = np.argsort(-self_influence_scores)
            return indices[:k]
        elif hardness == 'conf':
            self_influence_scores = trackin_self_influences['probs'][:, 0]
            indices = np.argsort(self_influence_scores)
            return indices[:k]
        else:
            raise NotImplementedError(f'{hardness} unknown')

    def get_least_k_influencial(trackin_self_influences, k=10):
        if hardness == 'tracin' or hardness == 'cscore':
            self_influence_scores = trackin_self_influences['self_influences']
            indices = np.argsort(self_influence_scores)
            return indices[:k]

        elif hardness == 'conf':
            self_influence_scores = trackin_self_influences['probs'][:, 0]
            indices = np.argsort(-self_influence_scores)
            return indices[:k]

        else:
            raise NotImplementedError(f'{hardness} unknonw')

    def load_self_influence_scores(run=0, path=None):

        if path is None:
            # save influence
            if aug_train_model_dir is not None:
                path = Path(aug_train_model_dir) / f'track_influences_dict_{run}'
            else:
                path = Path(dir_name) / f'track_influences_dict_{run}'

            with open(path, 'rb') as f:
                si = pickle.load(f)

        else:
            with open(path, 'rb') as f:
                si = pickle.load(f)
                import shutil
                shutil.copyfile(path, Path(dir_name) / f'track_influences_dict_{run}')

        print(si['self_influences'])
        return si

    def fit(dataset, dataset_val, train_images, test_images,
            save_checkpoints=True, log_dir=None, is_prediction_callback=True, early_stopping=True, memorisation_callback=False, run=0):

        if fit_seed:
            set_seed(seed=seed)

        prediction_callback = PredictionCallback(
            dataset, dataset_val, train_images,
            test_images, log_dir, binary=False)

        # show best/worst predictions during training
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_freq='epoch',
            verbose=1)

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=patience,
            verbose=1,
            mode="auto",
            baseline=None,
            restore_best_weights=restore_best_weights,
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1, profile_batch=0)

        callbacks = []
        if save_checkpoints:
            callbacks.append(model_checkpoint_callback)
        if is_prediction_callback:
            callbacks.append(prediction_callback)
        if early_stopping:
            callbacks.append(es)
        if memorisation_callback:
            mc = MemorisationCallback(run=run)
            callbacks.append(mc)

        callbacks.append(tensorboard_callback)

        model = get_model()
        model.summary()
        
        if optimizer_type == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                                            momentum=0.9)        
        elif hardness == 'conf' or optimizer_type == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        else:
            raise NotImplementedError(f'{optimizer_type} unknown')

        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=['accuracy'])

        history = model.fit(dataset,
                            epochs=epochs,
                            validation_data=dataset_val,
                            callbacks=callbacks)

        return model, history

    def find_high_low_influence_example(run=0, track_influences=None):

        log_dir = log_dir_base + f'/{run}/find_high_low_influence_example'

        (train_images, train_labels, test_images,
         test_labels) = get_dataset()

        n = 10
        msg = f"{n} training images"
        show_images(np.concatenate([train_images[:n]]),
                    log_dir=log_dir,
                    message=msg
                    )
        msg = f"{n} validation images"
        show_images(test_images[:n],
                    log_dir=log_dir,
                    message=msg)

        # preprocess
        print('Create tf datasets')
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        train_images = normalise(train_images, mean, std)
        test_images = normalise(test_images, mean, std)

        dataset_unbatched = tf.data.Dataset.from_tensor_slices(
            (train_images, train_labels))
        dataset = dataset_unbatched.batch(batch_size)
        dataset_val = tf.data.Dataset.from_tensor_slices(
            (test_images, test_labels))
        dataset_val = dataset_val.batch(batch_size)

        def train():

            model, history = fit(dataset, dataset_val, train_images, test_images,
                                 save_checkpoints=True, is_prediction_callback=False, early_stopping=True, log_dir=log_dir)

            # persist model and history

            # save history
            path = Path(dir_name) / f'train_history_dict_{run}'
            with open(path, 'wb') as file_pi:
                pickle.dump(history.history, file_pi)

            # save model weights
            path = Path(dir_name) / f'trained_model_{run}'
            model.save_weights(path.as_posix())

            return

        if track_influences is None:
            train()
            if trial_train:
                return
            track_influences = measure_influence(
                dataset_unbatched, train_images, train_labels, var=None)

        class_names = get_class_names()

        if hardness == 'tracin' or hardness == 'cscore':
            show_self_influence(track_influences,
                                train_images,
                                class_names,
                                topk=20,
                                var=None,
                                labels=train_labels,
                                log_dir=log_dir)

        elif hardness == 'conf':
            show_hardness(track_influences, train_images, topk=20, log_dir=log_dir)
        
        else:
            raise NotImplementedError(f'{hardness} uknown')

        # save influence
        path = Path(dir_name) / f'track_influences_dict_{run}'
        with open(path, 'wb') as file_pi:
            pickle.dump(track_influences, file_pi)

        return
    
    def setup_train_with_augmentation_on_index_tf(indx, run=0, artifact=None, log_dir=None):
        ds = get_dataset(indx, artifacter.artifact, batch_size=batch_size, log_dir=log_dir)
        return ds

    def set_batch_position(train_images, train_labels, indx):
        n_indx = batch_size * batch_position
        train_images[n_indx], train_images[indx] = train_images[indx], train_images[n_indx] 
        train_labels[n_indx], train_labels[indx] = train_labels[indx], train_labels[n_indx] 
        return train_images, train_labels, n_indx

    def setup_train_with_augmentation_on_index(indx, run=0, artifact=None, log_dir=None):

        ds = get_dataset()

        (train_images, train_labels, test_images,
            test_labels) = ds

        # manipulate the batch that the unique feature appears. this generates NEW canary index.
        if batch_position is not None:
            train_images, train_labels, indx = set_batch_position(train_images, train_labels, indx)

        # re-scale 0-1
        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0

        # apply augmentation
        if not baseline:
            train_images[indx] = artifacter.augment_image(
                train_images[indx], artifacter.artifact)

        # log augmented image
        show_image(train_images[indx], log_dir=log_dir,
                message=f'augmented (pre-processed) i={indx}, label={train_labels[indx]}')

        # normalise images after applying unique feature.
        train_images = normalise(train_images, mean, std)
        test_images = normalise(test_images, mean, std)

        # log augmented processed image
        show_image(train_images[indx], log_dir=log_dir,
                message=f'augmented (post-process) i={indx}, label={train_labels[indx]}')

        print('Create tf datasets')
        dataset = tf.data.Dataset.from_tensor_slices(
            (train_images, train_labels))
        dataset_val = tf.data.Dataset.from_tensor_slices(
            (test_images, test_labels))

        dataset = dataset.batch(batch_size)
        dataset_val = dataset_val.batch(batch_size)
    
        return dataset, dataset_val

    def train_with_augmentation_on_index(indx, run=0, artifact=None):

        log_dir = log_dir_base + f'/{run}/train_with_augmentation_on_index'

        if dataset_name in tf_datasets:
            ds = setup_train_with_augmentation_on_index_tf(indx, run=0, artifact=None, log_dir=log_dir)
        else:
            ds = setup_train_with_augmentation_on_index(indx, run=0, artifact=None, log_dir=log_dir)

        dataset, dataset_val = ds

        model, history = fit(dataset, dataset_val, None, None,
                             save_checkpoints=False, log_dir=log_dir, is_prediction_callback=False, memorisation_callback=run_memorisation_per_epoch, run=run)

        # persist

        # save history
        path = Path(dir_name) / f'aug_train_history_dict_{run}'
        with open(path, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        # save model weights
        path = Path(dir_name) / f'aug_trained_model_{run}'
        model.save_weights(path.as_posix())

        return model
    
    def get_aug_model(run):
        # run is the model run for the chosen canary.
        # we restore from the chosen trained model.
        # i.e. the one that respects the early stopping criteria
        if aug_train_model_dir is not None:
            path = Path(aug_train_model_dir) / f'aug_trained_model_{run}'
        else:
            path = Path(dir_name) / f'aug_trained_model_{run}'
        
        print(f'Load model from: {path}')
        
        tf.keras.backend.clear_session()
        model = get_model()
        model.load_weights(path.as_posix()).expect_partial()
        
        return model
    
    def get_ground_truth_label(indx):
        # indx: canary index
        # get the canary ground truth label
        train_images, train_labels, _, _ = get_dataset()
        gt_label = train_labels[indx]
        
        # come back to this.

        # manipulate the batch that the unique feature appears in
        if batch_position is not None:
            set_batch_position(train_images, train_labels, indx)

        return gt_label
    
    def get_ood_labels():
        _, labels, _, _ = get_ood_dataset()
        return labels

    def eval_with_augmentation_on_index(indx, run=0):
        # indx: canary index
        # run: model run for that canary

        # get model outputs/features
        gt_label = get_ground_truth_label(indx)
        labels = get_ood_labels()
        print('Eval on OOD Clean data')
        eval_1 = eval_ood(run)
        print('Eval on OOD Feature data')
        eval_2 = eval_ood(run, artifact_name='feature')
        print('Eval on OOD Random data')
        eval_3 = eval_ood(run, artifact_name='random')

        # model outputs
        outputs_1 = eval_1['outputs']
        outputs_2 = eval_2['outputs']
        outputs_3 = eval_3['outputs']

        # activations at each layer (could be None)
        activations_1 = eval_1['activations']
        activations_2 = eval_2['activations']
        activations_3 = eval_3['activations']

        # calculate metrics
        from scipy import stats
        
        # black box metric
        kl_1, kl_2 = metric_2(outputs_1, outputs_2, outputs_3)
        kl_1_2, kl_2_2 = kl_1.mean(), kl_2.mean()
        # pair two tail for means
        m2 = kl_1_2 - kl_2_2

        # white box metric
        kl_1, kl_2 = metric_3_alt(outputs_1, outputs_2, outputs_3, gt_label, labels)
        r = stats.ttest_rel(kl_2, kl_1, alternative='greater')
        m3_ttest_pvalue = r.pvalue
        m3_ttest_stat = r.statistic
        r = stats.mannwhitneyu(kl_2, kl_1, alternative='greater')
        m3_mannwu_pvalue = r.pvalue
        m3_mannwu_stat = r.statistic

        kl_1_3, kl_2_3 = kl_1.mean(), kl_2.mean()
        # pair two tail for means
        m3 = kl_1_3 - kl_2_3

        # some other metrics
        m7 = metric_7(outputs_2, outputs_3)
        m9 = metric_9(outputs_2, outputs_3)
        m10 = metric_10(outputs_2, outputs_3)

        # argmax softmax max unique feature metric
        m6 = metric_6(outputs_1, outputs_2, outputs_3, gt_label)
        
        m12 = metric_12(outputs_1, outputs_2, outputs_3, gt_label)
        m13 = metric_13(outputs_1, outputs_2, outputs_3, gt_label)
        m14 = metric_14(outputs_1, outputs_2, outputs_3, gt_label)
        m15 = metric_15(outputs_1, outputs_2, outputs_3, gt_label)

        # average entropy of the predictions
        m11_clean = metric_11(outputs_1)
        m11_unique = metric_11(outputs_2)
        m11_random = metric_11(outputs_3)

        return (m2, m3, m6, m7, m9, m10, m11_clean,
            m11_unique, m11_random, m12, m13, m14, m15, gt_label,
            kl_1_2, kl_2_2, kl_1_3, kl_2_3,
            activations_1, activations_2, activations_3,
            m3_ttest_pvalue, m3_ttest_stat,
            m3_mannwu_pvalue, m3_mannwu_stat)


    def eval_ood(run, artifact=None, artifact_name=None):

        model = get_aug_model(run)

        log_dir = log_dir_base + f'/eval_ood/0'

        try:
            train_images, _, _, _ = get_ood_dataset(ni=ni, nj=nj, seed=seed)
        except TypeError:
            print('Handling Request for non-dynamic OOD dataset')
            train_images, _, _, _ = get_ood_dataset()

        train_images = train_images / 255.0

        if (artifact_name == 'random') or (artifact_name == 'feature'):
            train_images = artifacter.augment_images(train_images, artifacter.artifact, feature_type=artifact_name)
            show_images(train_images[:10], log_dir=log_dir,
                        message=f'artifact {artifact_name}')
        else:
            show_images(train_images[:10], log_dir=log_dir,
                        message='artifact clean')

        normalise(train_images, mean, std)

        ds = tf.data.Dataset.from_tensor_slices(train_images)
        ds.batch(batch_size)

        outputs = model.predict(train_images)

        # save the activations at each layer of the model
        # only tested for MLP
        if save_activations:
            from utils import get_activations
            activations = get_activations(model, train_images)
        else:
            activations = None

        o = {
            'outputs': outputs,
            'activations': activations
        }

        return o

    def evaluate(epoch='last', indxs=None, mode='w'):

        print('Start Evaluate')
        
        count = 0

        # start logs from scratch
        paths = glob.glob(f"logs/fit/{exp_name}/*/eval_ood")
        for p in paths:
            shutil.rmtree(p)
        
        # pd data frame dict
        data = {
            'M2': [],
            'M3': [],
            'M6': [],
            'M7': [],
            'M9': [],
            'M10': [],
            'M11_clean': [],
            'M11_unique': [],
            'M11_random': [],
            'M12': [],
            'M13': [],
            'M14': [],
            'M15': [],
            'ground_truth_label': [],
            'influence': [],
            'canary': [],
            'p_time': [],
            'epoch': [],
            'kl_1_2': [],
            'kl_2_2': [],
            'kl_1_3': [],
            'kl_2_3': [],
            'm3_ttest_pvalue': [],
            'm3_ttest_stat': [],
            'm3_mannwu_pvalue': [],
            'm3_mannwu_stat': [],
        }

        # artifact 1, k best
        for indx in indxs:
            print(f'Eval canary {count}')
            t0 = time.perf_counter()
            (m2, m3, m6, m7, m9, m10, m11_clean,
            m11_unique, m11_random, m12, m13, m14, m15, gt_label,
            kl_1_2, kl_2_2, kl_1_3, kl_2_3,
            activations_1, activations_2, activations_3,
            m3_ttest_pvalue, m3_ttest_stat,
            m3_mannwu_pvalue, m3_mannwu_stat) = eval_with_augmentation_on_index(
                indx, run=count)

            data['M2'].append(m2)
            data['M3'].append(m3)
            data['M6'].append(m6)
            # means
            data['kl_1_2'].append(kl_1_2)
            data['kl_2_2'].append(kl_2_2)
            data['kl_1_3'].append(kl_1_3)
            data['kl_2_3'].append(kl_2_3)
            data['M7'].append(m7)
            data['M9'].append(m9)
            data['M10'].append(m10)
            data['M11_clean'].append(m11_clean)
            data['M11_unique'].append(m11_unique)
            data['M11_random'].append(m11_random)
            data['M12'].append(m12)
            data['M13'].append(m13)
            data['M14'].append(m14)
            data['M15'].append(m15)
            data['canary'].append(indx)
            data['ground_truth_label'].append(gt_label)
            data['influence'].append('random')
            data['p_time'].append(time.perf_counter() - t0)
            data['epoch'].append(epoch)
            data['m3_ttest_pvalue'].append(m3_ttest_pvalue)
            data['m3_ttest_stat'].append(m3_ttest_stat)
            data['m3_mannwu_pvalue'].append(m3_mannwu_pvalue)
            data['m3_mannwu_stat'].append(m3_mannwu_stat)

            if save_activations:
                np.save(f'{dir_name}/activations_clean_{count}', activations_1, allow_pickle=True)
                np.save(f'{dir_name}/activations_unique_{count}', activations_2, allow_pickle=True)
                np.save(f'{dir_name}/activations_random_{count}', activations_3, allow_pickle=True)

            count += 1
        df = pd.DataFrame(data=data)

        path = dir_name + '/eval.csv'
        if mode == 'a':
            df.to_csv(path, mode='a', header=not os.path.exists(path))
        else:
            df.to_csv(path)
        print(f'write eval to: {path}')

    def main():

        if run_find_si:
            find_high_low_influence_example()
        
        indxs = canary_indices

        if run_train_leaky_models:
            for i, indx in enumerate(indxs):
                print(f'Train canary {i}')
                train_with_augmentation_on_index(
                    indx, run=i, artifact=artifacter.artifact)

        if run_eval:
            evaluate(indxs=indxs)

    main()
