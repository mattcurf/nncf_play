#!/bin/bash
source /root/miniforge3/bin/activate ov
source /opt/intel/openvino_2024.4.0/setupvars.sh

cd /tmp
python3 /project/nncf-quant.py

echo "CPU fp32"
/root/openvino_cpp_samples_build/intel64/Release/benchmark_app -d CPU -niter 1000 -m output/resnet50_fp32.xml | grep Throughput
echo "CPU int8"
/root/openvino_cpp_samples_build/intel64/Release/benchmark_app -d CPU -niter 1000 -m output/resnet50_int8.xml | grep Throughput

if [ "$HAS_GPU" == "1" ]; then
  echo "GPU fp32"
  /root/openvino_cpp_samples_build/intel64/Release/benchmark_app -d GPU -niter 1000 -m output/resnet50_fp32.xml | grep Throughput
  echo "GPU int8"
  /root/openvino_cpp_samples_build/intel64/Release/benchmark_app -d GPU -niter 1000 -m output/resnet50_int8.xml | grep Throughput
fi

if [ "$HAS_NPU" == "1" ]; then
  # Workaround for defect https://github.com/intel/linux-npu-driver/issues/44
  sed -i "s/type=\"MaxPool\" version=\"opset14\"/type=\"MaxPool\" version=\"opset8\"/g"  output/resnet50_fp32.xml
  sed -i "s/type=\"MaxPool\" version=\"opset14\"/type=\"MaxPool\" version=\"opset8\"/g"  output/resnet50_int8.xml

  echo "NPU fp32"
  /root/openvino_cpp_samples_build/intel64/Release/benchmark_app -d NPU -niter 1000 -m output/resnet50_fp32.xml | grep Throughput
  echo "NPU int8"
  /root/openvino_cpp_samples_build/intel64/Release/benchmark_app -d NPU -niter 1000 -m output/resnet50_int8.xml | grep Throughput
fi
