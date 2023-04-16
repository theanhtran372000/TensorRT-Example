#!/bin/bash
CUDA_MODULE_LOADING=LAZY python3 inference.py \
    --engine_path=results/resnet50.engine \
    --input_path=resources/turkish_coffee.jpg \
    --class_path=resources/imagenet_classes.txt \
    --batch_size=2