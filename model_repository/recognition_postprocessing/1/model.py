import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(model_config, "recognition_postprocessing_output")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):

        responses = []
        def decodeText(scores):
            text = ""
            alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
            for i in range(scores.shape[0]):
                c = np.argmax(scores[i])
                if c != 0:
                    text += alphabet[c - 1]
                else:
                    text += "-"

            # adjacent same letters as well as background text must be removed to get the final output
            char_list = []
            for i in range(len(text)):
                if text[i] != "-" and (not (i > 0 and text[i] == text[i - 1])):
                    char_list.append(text[i])
            return "".join(char_list)

        for request in requests:
            
            in_1 = pb_utils.get_input_tensor_by_name(request, "recognition_postprocessing_input").as_numpy()
            text_list = []
            
            for i in range(in_1.shape[0]):
                text_list.append(decodeText(in_1[i]))
            print(text_list, flush=True)
            
            output = pb_utils.Tensor("recognition_postprocessing_output", np.array(text_list).astype(self.output_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print("Cleaning up...")