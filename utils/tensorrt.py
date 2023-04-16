import time
import tensorrt as trt

def build_serialized_engine(
    logger, 
    onnx_file_path, 
    save_path, 
    input_name='input', 
    input_shape=(3, 224, 224),
    min_batch=1, 
    opt_batch=8, 
    max_batch=32, 
    workspace_size=1):
    
    start = time.time()
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    # Initialize TensorRT engine and parse ONNX model
    with trt.Builder(logger) as builder, \
        builder.create_network(explicit_batch_flag) as network, \
        builder.create_builder_config() as config:
        
        # Parse ONNX
        parser = trt.OnnxParser(network, logger)
        with open(onnx_file_path, 'rb') as model:
            print('Begin parsing ONNX file')
            parser.parse(model.read())
        print('Completed parsing ONNX model')

        # Config builder
        config = builder.create_builder_config()
        # allow TensorRT to use up to 1GB of GPU memory for tactic selection
        if workspace_size != 0:
            # TensorRT >= 8.5
            # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size << 30) 
            pass
        # use FP16 mode if possible
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        # Add optimization profile for 'input' (required when using dynamic batch size)
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, (min_batch, *input_shape), (opt_batch, *input_shape), (max_batch, *input_shape))
        config.add_optimization_profile(profile)

        # generate TensorRT engine optimized for the target platform
        print('Building an engine...')
        serialized_engine = builder.build_serialized_network(network, config)

        print("Completed creating Engine after {:.2f}s".format(time.time() - start))
        
        with open(save_path, 'wb') as f:
            f.write(serialized_engine)
 
        print('Write serialize engine to {}!'.format(save_path))

   
# Deserialized engine
def build_deserialized_engine(logger, engine_path):
    
    # Load saved engine
    with open (engine_path, 'rb') as f:
        serialized_engine = f.read()
    
    # Deserialize engine
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    return engine

# For TensorRT >= 8.5
def create_context_v2(engine, stream_handle, input_shape, input_name='input'):
    context = engine.create_execution_context()
    context.set_optimization_profile_async(0, stream_handle)
    context.set_input_shape(input_name, trt.Dims4(list(input_shape)))
    return context

# For TensorRT 8.2
def create_context_v1(engine, stream_handle, input_shape, binding=0):
    context = engine.create_execution_context()
    context.set_optimization_profile_async(0, stream_handle)
    context.set_binding_shape(binding, trt.Dims4(list(input_shape)))
    return context


def get_logger():
    # Logger to capture errors, warnings and other infomation
    TRT_LOGGER = trt.Logger()
    
    return TRT_LOGGER