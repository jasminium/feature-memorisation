# Unintended memorisation of unique features in neural networks

by John Hartley and Sotirios A. Tsaftaris

[paper here]()

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
