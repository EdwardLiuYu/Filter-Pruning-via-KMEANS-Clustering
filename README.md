# Filter-Pruning-via-KMEANS-Clustering
Pytorch implementation for "More similar Less Important: Filter Pruning via KMeans Clustering"
[IEEE ICME2021](https://ieeexplore.ieee.org/document/9428286)

This implementation is based on [filter-pruning-geometric-median](https://github.com/he-y/filter-pruning-geometric-median).

## Requirements
- Python 3.6
- PyTorch 0.3.1
- TorchVision 0.3.0

## Training ResNet on ImageNet

#### Usage of Pruning Training

We train each model from scratch with stepwise learning rate decay by default. If you wish to train the model with pre-trained models and cosine annealing learning rate decay, please use the options `--use_pretrain --lr 0.01 --cos`

Run Pruning Training ResNet (depth 101,50,34,18) on Imagenet:
```bash
python pruning_kmeans_imagenet.py -a resnet101 --save_dir ./snapshots/resnet101_8_04 --pruning_rate 0.4 --n_clusters 8 --layer_begin 0 --layer_end 309 --layer_inter 3  /path/to/Imagenet2012

python pruning_kmeans_imagenet.py -a resnet50  --save_dir ./snapshots/resnet50_8_04 --pruning_rate 0.4 --n_clusters 8 --layer_begin 0 --layer_end 156 --layer_inter 3  /path/to/Imagenet2012

python pruning_kmeans_imagenet.py -a resnet34  --save_dir ./snapshots/resnet34_8_04 --pruning_rate 0.4 --n_clusters 8 --layer_begin 0 --layer_end 105 --layer_inter 3  /path/to/Imagenet2012

python ppruning_kmeans_imagenet.py -a resnet18  --save_dir ./snapshots/resnet18_8_04 --pruning_rate 0.4 --n_clusters 8 --layer_begin 0 --layer_end 57 --layer_inter 3  /path/to/Imagenet2012
```

## Training ResNet on CIFAR10

#### Usage of Pruning Training

```bash
sh scripts/pruning_cifar10_new.sh
```
