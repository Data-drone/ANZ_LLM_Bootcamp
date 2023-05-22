#!/bin/bash

apt-get install -y libaio-dev

# 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusparse-11-7_11.7.4.91-1_amd64.deb -O /tmp/libcusparse-11-7_11.7.4.91-1_amd64.deb
dpkg -i /tmp/libcusparse-11-7_11.7.4.91-1_amd64.deb

#
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusparse-dev-11-7_11.7.4.91-1_amd64.deb -O /tmp/libcusparse-dev-11-7_11.7.4.91-1_amd64.deb
dpkg -i /tmp/libcusparse-dev-11-7_11.7.4.91-1_amd64.deb

# 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcublas-11-7_11.10.3.66-1_amd64.deb -O /tmp/libcublas-11-7_11.10.3.66-1_amd64.deb
dpkg -i /tmp/libcublas-11-7_11.10.3.66-1_amd64.deb

# 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcublas-dev-11-7_11.10.3.66-1_amd64.deb -O /tmp/libcublas-dev-11-7_11.10.3.66-1_amd64.deb
dpkg -i /tmp/libcublas-dev-11-7_11.10.3.66-1_amd64.deb

# 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusolver-11-7_11.4.0.1-1_amd64.deb -O /tmp/libcusolver-11-7_11.4.0.1-1_amd64.deb
dpkg -i /tmp/libcusolver-11-7_11.4.0.1-1_amd64.deb

# 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-11-7_10.2.10.91-1_amd64.deb -O /tmp/libcurand-11-7_10.2.10.91-1_amd64.deb
dpkg -i /tmp/libcurand-11-7_10.2.10.91-1_amd64.deb

# 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-dev-11-7_10.2.10.91-1_amd64.deb -O /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb
dpkg -i /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb

#
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb -O /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb
dpkg -i /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb
