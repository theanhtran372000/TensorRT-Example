# TensorRT Example
A basic example of using TensorRT version 8.2 on Jetson Nano

## 1. Environment
- Ubuntu: 18.04
- CUDA: 10.2
- CuDNN: 8.2.1
- TensorRT: 8.2.1.8
- PyCUDA: 2022.1

## 2. Resources
- ONNX model: [Google Drive](https://drive.google.com/file/d/1zcZvyroSZcIgzNkFngEHdekY79ptEQxs/view?usp=share_link)
- TensorRT model: [Google Drive](https://drive.google.com/file/d/1cFqAf7VVBgRNb9McEJYkTDcA-Px4kvYh/view?usp=share_link)

## 3. Guidance
- Download and put ONNX model to /results folder
- Run
```
bash scripts/build.sh # Build TensorRT model
bash scripts/infer.sh # Run inference
```

## 4. References
- https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/