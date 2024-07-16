import torch
from networks.STR import STRModel

model = STRModel(input_channels=1, output_channels=512, num_classes=37)
state = torch.load("weights/None-ResNet-None-CTC.pth")
state = {key.replace("module.", ""): value for key, value in state.items()}
model.load_state_dict(state)

trace_input = torch.randn(1, 1, 32, 100)
dynamic_batching = True

if dynamic_batching:
    torch.onnx.export (
        model, trace_input, "/workspace/triton/model_repository/text_reg_batch/1/model.onnx", export_params=True, 
        input_names = ['input.1'], output_names = ['308'],
        dynamic_axes={'input.1': {0: 'batch_size'}, '308': {0: 'batch_size'}}
    )
    
else:
    torch.onnx.export (
        model, trace_input, "/workspace/triton/model_repository/text_recognition/1/model.onnx", export_params=True, 
        input_names = ['input.1'], output_names = ['308']
    )