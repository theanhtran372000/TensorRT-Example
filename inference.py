import argparse
import time
import numpy as np

import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from utils import build_deserialized_engine, get_logger, create_context_v1
from utils import preprocess_image, postprocess_result

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--engine_path', type=str, help='Path to built engine file')
    parser.add_argument('--input_path', type=str, help='Path to input image')
    parser.add_argument('--class_path', type=str, help='Path to class file')
    parser.add_argument('--batch_size', type=int, default=1, help='Inference batch size')
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Load and deserialize engine
    print('Loading and deserializing engine')
    engine = build_deserialized_engine(get_logger(), args.engine_path)
    
    # Prepare input with batch size
    batch_size = args.batch_size # Inference batch size
    sample_image = preprocess_image("resources/turkish_coffee.jpg").numpy()
    batch_images = np.concatenate([sample_image] * batch_size, axis=0)
    
    # TensorRT >= 8.5
    # Get sizes of input and output and allocate memory required for input and output data
    # for idx, _tensor in enumerate(engine): # inputs and outputs
    #     name = engine.get_tensor_name(idx)
        
    #     if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT: # in case it is input
    #         input_shape = engine.get_tensor_shape(_tensor)
    #         input_shape[0] = batch_size
            
    #         input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize # in bytes
    #         device_input = cuda.mem_alloc(input_size)
    #     else: # output
    #         output_shape = engine.get_tensor_shape(_tensor)
    #         output_shape[0] = batch_size
            
    #         # Create page-locked memory buffer (i.e. won't be swapped to disk)
    #         host_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
    #         device_output = cuda.mem_alloc(host_output.nbytes)
    
    # For TensorRT 8.2
    input_binding = -1
    for idx, binding in enumerate(engine): # inputs and outputs
        
        if engine.binding_is_input(binding): # in case it is input
            input_binding = idx
            input_shape = engine.get_binding_shape(binding)
            input_shape[0] = batch_size
            
            input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize # in bytes
            device_input = cuda.mem_alloc(input_size)
        else: # output
            output_shape = engine.get_binding_shape(binding)
            output_shape[0] = batch_size
            
            # Create page-locked memory buffer (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)
     
    # Create a stream in which to copy inputs/outputs and run inference
    stream = cuda.Stream()
    
    print('Inferencing')
    start = time.time()
    
    # Preprocess input data
    host_input = np.array(batch_images, dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)
    
    # Create context
    context = create_context_v1(engine, stream.handle, host_input.shape, input_binding)
    
    # Run inference
    context.execute_async_v2(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()
    
    # Post processing (return value in host_output is one dimension array)
    tensor_output = torch.Tensor(host_output)
    output_data = tensor_output.reshape(batch_size, int(tensor_output.shape[0] / batch_size))
    for i, output in enumerate(output_data):
        print('=== Image {} ==='.format(i))
        postprocess_result(output.reshape(1, -1), class_path=args.class_path)
    
    print('Complete inferencing batch {} samples after {:.4f}s'.format(batch_size, time.time() - start))

if __name__ == '__main__':
    main()