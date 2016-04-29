
# The base of this has come from http://ramhiser.com/2016/01/05/installing-tensorflow-on-an-aws-ec2-instance-with-gpu-support/

# Also https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#optional-install-cuda-gpus-on-linux

##
# When configuring tensorflow must use correct cuda cudnn versions which you downloaded.
##

# Update essentials
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install -y build-essential git python-pip libfreetype6-dev libxft-dev libncurses-dev libopenblas-dev gfortran python-matplotlib libblas-dev liblapack-dev libatlas-base-dev python-dev python-pydot linux-headers-generic linux-image-extra-virtual unzip python-numpy swig python-pandas python-sklearn unzip wget pkg-config zip g++ zlib1g-dev
sudo pip install -U pip


# Get cuda 
wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
rm cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda
# Restart


# Download cudnn
# https://developer.nvidia.com/rdp/cudnn-download
tar xvzf cudnn-7.5-linux-x64-v4.tgz
sudo cp cudnn-7.5-linux-x64-v4/include/cudnn.h /usr/local/cuda/include
sudo cp cudnn-7.5-linux-x64-v4/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*


# Update ~/.bashrc
export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
export PATH=$PATH:$CUDA_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64

# Install java
sudo add-apt-repository -y ppa:webupd8team/java
sudo apt-get update
sudo apt-get install -y oracle-java8-installer

# Install Bazel
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip
wget https://github.com/bazelbuild/bazel/releases/download/0.1.4/bazel-0.1.4-installer-linux-x86_64.sh
chmod +x bazel-0.1.4-installer-linux-x86_64.sh
./bazel-0.1.4-installer-linux-x86_64.sh --user

# Add source /home/ubuntu/.bazel/bin/bazel-complete.bash to ~/.bashrc

# Invalid path to cuDNN  toolkit. Neither of the following two files can be found:
# /usr/local/cuda-7.5/lib64/libcudnn.so
# /usr/local/cuda-7.5/libcudnn.so
