#!/bin/bash
CUDA_MODULE_LOADING=LAZY python /home/theanh/workspace/test/build_engine.py \
    --onnx_path=/home/theanh/workspace/test/results/resnet50.onnx \
    --engine_path=/home/theanh/workspace/test/results/resnet50.engine \
    --workspace_size=2