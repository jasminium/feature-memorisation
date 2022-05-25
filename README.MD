# Unintended memorisation of unique features in neural networks

by John Hartley and Sotirios A. Tsaftaris

[paper here]()

## Abstract

Neural networks pose a privacy risk due to their propensity to memorise and leak training data. We show that unique features occurring only once in training data are memorised by discriminative multi-layer perceptrons and convolutional neural networks trained on benchmark imaging datasets. We design our method for settings where sensitive training data is not available, for example medical imaging. Our setting knows the unique feature, but not the training data, model weights or the unique feature’s label. We develop a score estimating a model’s sensitivity to a unique feature by comparing the KL divergences of the model’s output distributions given modified out-of-distribution images. We find that typical strategies to prevent overfitting do not prevent unique feature memorisation. And that images containing a unique feature are highly influential, regardless of the influence the images’s other features. We also find a significant variation in memorisation with training seed. These results imply that neural networks pose a privacy risk to rarely occurring private information. This risk is more pronounced in healthcare applications since sensitive patient information can be memorised when it remains in training data due to an imperfect data sanitisation process.


## Installation

1. Clone the unintended memorisation repository

```Shell
git clone https://github.com/jasminium/feature-memorisation.git
```

2. Install dependencies

```shell
cd feature-memorisation
conda install -c conda-forge tensorflow
pip install matplotlib
pip install notebook
pip install Pillow
pip install seaborn
```

## Run

We provide a jupyter notebook in ```um/um.ipynb``` to train 100 MLP models on MNIST, and evaluate the memorisation scores for 100 canaries. We also provide the models and datasets used in the paper.

## Contact

John Hartley john.hartley@ed.ac.uk
