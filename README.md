# DAWNBench Models

This repository contains implementations for various models presented in the DAWNBench Leaderboard:
- ResNet models for CIFAR10, implemented in TensorFlow, located at
  [`tensorflow/CIFAR10`](https://github.com/stanford-futuredata/dawn-bench-models/tree/master/tensorflow/CIFAR10)
- ResNet models for CIFAR10, implemented in PyTorch, located at
  [`pytorch/CIFAR10`](https://github.com/stanford-futuredata/dawn-bench-models/tree/master/pytorch/CIFAR10)
- BiDAF model for SQuAD, implemented in TensorFlow, located at
  [`tensorflow/SQuAD`](https://github.com/stanford-futuredata/dawn-bench-models/tree/master/tensorflow/SQuAD)

# Installation and setup

These steps explain how to set up all the necessary packages and libraries to run the above three models on a system with Nvidia GPUs.

## 1. Install CUDA

```
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
sudo apt-get install cuda-9-0
```

## 2. Install cuDNN

Download cuDNN v7.0.5 for CUDA 9.0 runtime for Ubuntu 16.04 from https://developer.nvidia.com/rdp/cudnn-archive. Then run:

```
sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
```

## 3. Install python3

```
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get install python3.6
```

## 4. Set up python virtualenv

```
sudo pip3 install virtualenv
virtualenv dawnenv --python=python3.6
source ./dawnenv/bin/activate
pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl  # PyTorch 0.4.0
pip3 install -r requirements.txt  # Includes tensorflow-gpu 1.8.0
```

## 5. Download data sets

Download the CIFAR10 data set and copy it to the appropriate directories.

```
curl -o cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
mkdir tensorflow/CIFAR10/cifar10
cp cifar-10-batches-bin/*.bin tensorflow/CIFAR10/cifar10/
```

Download the SQuAD data set, preprocess it.

```
cd tensorflow/SQUaD
mkdir data
chmod +x download.sh; ./download.sh
export NLTK_DATA=./data/nltk_data/
python3 -m squad.prepro --source_dir=./data/squad --glove_dir=./data/glove
```

## 6. Run benchmarks

### Tensorflow/CIFAR10

#### Train

```
cd tensorflow/CIFAR10
python3 resnet/resnet_main.py --train_data_path=cifar10/data_batch* --log_root=data/resnet20/log_root \
                              --train_dir=data/resnet20/log_root/train --dataset='cifar10' --model=resnet20 \
                              --num_gpus=1 --checkpoint_dir=data/resnet20/checkpoints --data_format=NCHW
```

#### Test

```
cd tensorflow/CIFAR10
python3 eval_checkpoints.py -i data/resnet20/checkpoints \
                            -c "python3 resnet/resnet_main.py --mode=eval --eval_data_path=cifar10/test_batch.bin --eval_dir=data/resnet20/log_root/eval --dataset='cifar10' --model=resnet20 --num_gpus=1 --eval_batch_count=100 --eval_once=True --data_format=NCHW"
```

### Tensorflow/SQuAD

#### Train

```
cd tensorflow/SQuAD
python3 -m basic.cli --mode train --noload --len_opt --cluster
```

### Test

```
cd tensorflow/SQuAD
python3 -m basic.cli --len_opt --cluster
```

### PyTorch/CIFAR10

This model needs more testing and may need more tweaks to properly PyTorch 0.4.0. Right now only training is functioning.

#### Train

```
cd pytorch/CIFAR10
pip3 install -e .
cifar10 train -c last --augmentation --tracking -b 128 --optimizer sgd --arch preact164 -e 5 -l 0.01
cifar10 train -c last --augmentation --tracking -b 128 --optimizer sgd --arch preact164 -e 90 -l 0.1 --restore latest
cifar10 train -c last --augmentation --tracking -b 128 --optimizer sgd --arch preact164 -e 45 -l 0.01 --restore latest
cifar10 train -c last --augmentation --tracking -b 128 --optimizer sgd --arch preact164 -e 45 -l 0.001 --restore latest
```
