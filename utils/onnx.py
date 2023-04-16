def to_onnx(
    model,
    fake_input,
    onnx_path,
    dynamic_batches=True):
    
    if dynamic_batches:
        dynamic_axes = {
            "input":{
                0: "batch_size"
            }, 
            "output":{
                0: "batch_size"
            }
        }

        torch.onnx.export(
            model,
            fake_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            dynamic_axes=dynamic_axes
        )
    else:
        torch.onnx.export(
            model,
            fake_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            export_params=True
        )