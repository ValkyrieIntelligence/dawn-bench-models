# DAWNBench Models

This repository contains implementations for various models presented in the DAWNBench Leaderboard:
- ResNet models for CIFAR10, implemented in TensorFlow, located at
  [`tensorflow/CIFAR10`](https://github.com/stanford-futuredata/dawn-bench-models/tree/master/tensorflow/CIFAR10)
- ResNet models for CIFAR10, implemented in PyTorch, located at
  [`pytorch/CIFAR10`](https://github.com/stanford-futuredata/dawn-bench-models/tree/master/pytorch/CIFAR10)
- BiDAF model for SQuAD, implemented in TensorFlow, located at
  [`tensorflow/SQuAD`](https://github.com/stanford-futuredata/dawn-bench-models/tree/master/tensorflow/SQuAD)

You can email us at [dawn-benchmark@lists.stanford.edu](mailto:dawn-benchmark@lists.stanford.edu) with any
questions.

# Installation and setup

These steps explain how to set up all the necessary packages and libraries to run the above three models on a system with Nvidia GPUs.

1. Install CUDA

```
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
sudo apt-get install cuda-9-0
```

2. Install cuDNN

Download cuDNN v7.0.5 for CUDA 9.0 runtime for Ubuntu 16.04 from https://developer.nvidia.com/rdp/cudnn-archive. Then run:

```
sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
```

3. Install python3

```
sudo apt install python3
```

4. Set up python virtualenv

5. Download data sets

Download the CIFAR10 data set and copy it to the appropriate directories.

```
curl -o cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
```

6. Run benchmarks


