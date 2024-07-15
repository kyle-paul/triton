import torch
from networks.STR import STRModel

model = STRModel(input_channels=1, output_channels=512, num_classes=37)
state = torch.load("weights/None-ResNet-None-CTC.pth")
state = {key.replace("module.", ""): value for key, value in state.items()}
model.load_state_dict(state)

trace_input = torch.randn(1, 1, 32, 100)
torch.onnx.export(model, trace_input, "model.onnx", 
                  export_params=True, 
                  input_names = ['input.1'],
                  output_names = ['308'])