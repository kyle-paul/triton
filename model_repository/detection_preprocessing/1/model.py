import io
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image

class TritonPythonModel:

    def initialize(self, args):

        model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(model_config, "detection_preprocessing_output")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):

        responses = []        
        for request in requests:
            input_image = pb_utils.get_input_tensor_by_name(request, "detection_preprocessing_input")

            def image_loader(image) -> np.ndarray:
                image = np.array(image.resize((640, 480)))
                image = image - np.array([103.94, 123.68, 116.78]).reshape(1,1,3)
                image = image.astype(np.float32)[None, ...]
                return image

            image = Image.open(io.BytesIO(input_image.as_numpy().tobytes()))
            image = image_loader(image)
            output = pb_utils.Tensor("detection_preprocessing_output", image.astype(self.output_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[output])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print("Cleaning up...")