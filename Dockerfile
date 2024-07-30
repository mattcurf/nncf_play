FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=america/los_angeles

# Base packages
RUN apt update && \
    apt install --no-install-recommends -q -y \
    build-essential \
    ocl-icd-libopencl1 \
    software-properties-common \
    python3 \
    python3-dev \
    libtbb12 \
    wget 

# Intel GPU compute user-space drivers
RUN mkdir -p /tmp/gpu && \
 cd /tmp/gpu && \
 wget https://github.com/oneapi-src/level-zero/releases/download/v1.17.19/level-zero_1.17.19+u22.04_amd64.deb && \
 wget https://github.com/oneapi-src/level-zero/releases/download/v1.16.1/level-zero_1.16.1+u22.04_amd64.deb && \
 wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17193.4/intel-igc-core_1.0.17193.4_amd64.deb && \
 wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17193.4/intel-igc-opencl_1.0.17193.4_amd64.deb && \
 wget https://github.com/intel/compute-runtime/releases/download/24.26.30049.6/intel-level-zero-gpu_1.3.30049.6_amd64.deb && \
 wget https://github.com/intel/compute-runtime/releases/download/24.26.30049.6/intel-opencl-icd_24.26.30049.6_amd64.deb && \
 wget https://github.com/intel/compute-runtime/releases/download/24.26.30049.6/libigdgmm12_22.3.20_amd64.deb && \
 dpkg -i *.deb && \
 rm *.deb

# Intel NPU compute user-space drivers
RUN mkdir -p /tmp/npu && \
  cd /tmp/npu && \
  wget https://github.com/oneapi-src/level-zero/releases/download/v1.17.2/level-zero_1.17.2+u22.04_amd64.deb && \
  wget https://github.com/intel/linux-npu-driver/releases/download/v1.5.0/intel-driver-compiler-npu_1.5.0.20240619-9582784383_ubuntu22.04_amd64.deb && \
  wget https://github.com/intel/linux-npu-driver/releases/download/v1.5.0/intel-level-zero-npu_1.5.0.20240619-9582784383_ubuntu22.04_amd64.deb && \
  dpkg -i *.deb && \
  rm *.deb

# OpenVINO via package
RUN mkdir -p /opt/intel && \
  wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.2/linux/l_openvino_toolkit_ubuntu22_2024.2.0.15519.5c0f38f83f6_x86_64.tgz --output-document openvino_2024.2.0.tgz && \
  tar xf openvino_2024.2.0.tgz && \
  rm openvino_2024.2.0.tgz && \
  mv l_openvino_toolkit_ubuntu22_2024.2.0.15519.5c0f38f83f6_x86_64 /opt/intel/openvino_2024.2.0 && \
  cd /opt/intel/openvino_2024.2.0  && \
  ./install_dependencies/install_openvino_dependencies.sh -y && \
  python3 -m pip install --upgrade pip && \
  python3 -m pip install -r ./python/requirements.txt torch torchvision nncf jupyter

RUN /bin/bash -c "source /opt/intel/openvino_2024.2.0/setupvars.sh && \
  /opt/intel/openvino_2024.2.0/samples/cpp/build_samples.sh"
 

