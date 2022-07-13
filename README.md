# How to Train Your Wide Neural Network Without Backprop: An Input-Weight Alignment Perspective

## Overview
This repository contains the code for "[How to Train Your Wide Neural Network Without Backprop: An Input-Weight Alignment Perspective](https://arxiv.org/abs/2106.08453)."  In this work, we explore the intersection of Neural Tangent Kernel (NTK) theory and biologically plausible neural networks. We show theoretically that gradient descent drives layerwise weight updates that are aligned with their input activity correlations weighted by error, and demonstrate empirically that the result also holds in finite-width wide networks. The alignment result allows us to formulate a family of biologically-motivated, backpropagation-free learning rules that are theoretically equivalent to backpropagation in infinite-width networks. We test these learning rules on benchmark problems in feedforward and recurrent neural networks and demonstrate, in wide networks, comparable performance to backpropagation. The proposed rules are particularly effective in low data regimes, which are common in biological learning settings.

## Prerequisites & Setup
This repository requires the following Python libraries: Tensorflow, Numpy, PyTorch and Torchvision. 

To set up the necessary environment, please follow these steps:

1. Clone this repository:
```
git clone https://github.com/FieteLab/Wide-Network-Alignment
cd Wide-Network-Alignment
```

2. Required packages are included in `requirements.txt`. They can be installed with pip via:
```
pip install -r requirements.txt
```

3. For experiments on ImageNet, move ImageNet train and validation directories to the directory in which the repository is located:
```
cd ..
mkdir train
mkdir val
mv /path/to/imagenet/train/* train
mv /path/to/imagenet/val/* val
```

## How to Run

For all experiments, results will be printed to standard output.

### Alignment score experiments
To run experiments computing alignment scores, first train the necessary networks with:
```
python3 alignment_<dataset>.py 1
```
where `<dataset>` is replaced by `cifar` or `kmnist` for CIFAR-10 or KMNIST respectively. 

To compute alignment scores for networks with varying widths, run:
```
python3 alignment_<dataset>.py 2
```

To compute alignment scores for networks for different amounts of training, run:
```
python3 alignment_<dataset>.py 3
```
where `<dataset>` is replaced by `cifar`, `kmnist`, `add` or `imagenet` for CIFAR-10, KMNIST, Add task or ImageNet respectively. 


### Align learning rule experiments
To run experiments comparing the performance of Align learning rules with baselines, run:
```
python3 train_<dataset>.py <number>
```
where `<dataset>` is replaced by `cifar`, `kmnist`, `add` or `imagenet` for CIFAR-10, KMNIST, Add task or ImageNet respectively. 

For ImageNet and Add task experiments, replace `<number>` to run all experiments. For CIFAR-10 and KMNIST, setting `<number>` to `1` will run experiments comparing networks of varying networks widths, while `2` will run experiments comparing networks trained at different learning rates. CIFAR-10 has the following additional experiments: `3`: standard parameterization experiments, `4`: Small CIFAR-10 experiments, `5`: Align-prop experiments, `6`: seed robustness experiments, `7`: training time experiments.


## Files

`setup_*.py` contain code to load CIFAR-10 and KMNIST data.

`alignment_*.py` contain code to compute alignment scores of networks trained on CIFAR-10 and KMNIST.

`train_*.py` contain code to train networks with Align methods and baselines on the Add task, CIFAR-10, KMNIST and ImageNet.

`outputs/` will contain parameters of networks necessary to compute alignment scores.

## Citation
If you find this work useful, we would appreciate if you could cite our paper:
```
@inproceedings{boopathy2022how,
    author = {Boopathy, Akhilan and Fiete, Ila},
    title = {How to Train Your Wide Neural Network Without Backprop: An Input-Weight Alignment Perspective},
    booktitle = {ICML},
    year = {2022},
}   
```
