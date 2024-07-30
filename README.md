# NNCF Play

This repo demonostrates model quantization using [nncf](https://github.com/openvinotoolkit/nncf), referencing the documentation at https://docs.openvino.ai/2023.3/notebooks/112-pytorch-post-training-quantization-nncf-with-output.html to demonstrate the performance benefit and accuracy tradeoff with the resnet50 model

# Prerequisites
* Ubuntu 24.04 or newer (for Intel ARC GPU kernel driver support. Tested with Ubuntu 24.04), or Windows 11 with WSL2 (graphics driver [101.5445](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html) or newer)
* Installed Docker and Docker-compose tools (for Linux) or Docker Desktop (for Windows)
* Intel ARC series GPU (tested with Intel ARC A770 16GB and Intel(R) Core(TM) Ultra 5 125H integrated GPU)

# Docker 

These samples utilize containers to fully encapsulate the example with minimial host dependencies.  Here are the instructions how to install docker:

```
$ sudo apt-get update
$ sudo apt-get install ca-certificates curl
$ sudo install -m 0755 -d /etc/apt/keyrings
$ sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
$ sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install docker
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Enable docker access as user
sudo groupadd docker
sudo usermod -aG docker $USER
```

## Usage

To build the container, type:
```
$ ./build
```

To download the PyTorch Resnet50 model, quantize the fp32 model to int8 using nncf, perform an accuracy analysis between the two models, and then perform a performance benchmark on the two models using CPU and GPU, type:
```
$ ./run
```

