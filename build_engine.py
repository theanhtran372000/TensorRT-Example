import argparse
from utils import build_serialized_engine, get_logger

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--onnx_path', type=str, help='Path to ONNX file')
    parser.add_argument('--engine_path', type=str, help='Path to save engine')
    parser.add_argument('--workspace_size', type=int, default=2, help='Size in GB use for temporary storing')
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    logger = get_logger()

    # Build engine
    build_serialized_engine(
        logger,
        args.onnx_path,
        args.engine_path,
        input_name='input',                 # input name in ONNX file
        input_shape=(3, 224, 224),          # image shape
        min_batch=1,                        # min inference batch
        opt_batch=8,                        # optimized (best) inference batch
        max_batch=32,                       # max inference batch
        workspace_size=args.workspace_size, # available work space
    )
    
if __name__ == '__main__':
    main()