import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.grpc import service_pb2, service_pb2_grpc
import time

class TritonClient:
    def __init__(self):
        self.url = "localhost:8001"
        self.client = grpcclient.InferenceServerClient(url=self.url)
        
    def infer(self, model_name, input_data):
        inputs = []
        input_tensor = grpcclient.InferInput("input", input_data.shape, "FP32")
        input_tensor.set_data_from_numpy(input_data)
        inputs.append(input_tensor)
        
        outputs = []
        output_tensor = grpcclient.InferRequestedOutput("output")
        outputs.append(output_tensor)
        
        start = time.time()
        response = self.client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        print("Model inference time", (time.time() - start)*1000, "ms")
        
        output_data = response.as_numpy("output")
        return output_data

        
if __name__ == "__main__":
    model_name = "classification"
    input_data = np.zeros((1, 3, 224, 224), dtype=np.float32)
    
    triton_client = TritonClient()
    start = time.time()
    output = triton_client.infer(model_name, input_data)
    print("Total call inference time", (time.time() - start)*1000, "ms")
    print(output.shape)