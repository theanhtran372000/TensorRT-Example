#!/bin/bash
CUDA_MODULE_LOADING=LAZY python3 build_engine.py \
    --onnx_path=results/resnet50.onnx \
    --engine_path=results/resnet50.engine \
    --workspace_size=1