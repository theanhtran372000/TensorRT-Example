#!/bin/bash
CUDA_MODULE_LOADING=LAZY python /home/theanh/workspace/test/inference.py \
    --engine_path=/home/theanh/workspace/test/results/resnet50.engine \
    --input_path=/home/theanh/workspace/test/resources/turkish_coffee.jpg \
    --class_path=/home/theanh/workspace/test/resources/imagenet_classes.txt \
    --batch_size=2