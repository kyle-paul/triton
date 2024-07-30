from fastapi import FastAPI, File, UploadFile, HTTPException
from enum import Enum
from PIL import Image
import numpy as np
from io import BytesIO
from client.classification_grpc import TritonClient

app = FastAPI()

@app.get("/")
def read_root():
    return {"Deep Learning": "Computer Vision"}

class ModelName(str, Enum):
    classification = "classification"
    mobilenet = "mobilenet"
    resnet = "resnet"
    googlenet = "googlenet"
    
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = image[np.newaxis, :, :, :]
    return image.astype(np.float32)

@app.post("/models/{model_name}")
async def infer_model(model_name: ModelName, file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    input_data = preprocess_image(image)
    
    triton_client = TritonClient()
    output = triton_client.infer(model_name=model_name.value, input_data=input_data)
    
    return {"output_shape": output.shape}
