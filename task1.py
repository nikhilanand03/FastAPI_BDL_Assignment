from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from PIL import Image 
import PIL
import numpy as np
import io
# import Request

app = FastAPI()

@app.get("/")
async def root():
    return {}

# @app.post("/putObject")
# async def put_object(request: Request, application: str, file: UploadFile) -> str:

#         request_object_content = await f2ile.read()
#         img = Image.open(io.BytesIO(request_object_content))

@app.post("/predict")
async def predict(file: UploadFile):
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content)) 

    arr = np.array(img)
    
    print(arr,arr.shape)

    flattened_image=arr.reshape(-1)
    flattened_image_list = flattened_image.tolist()
   
    model = await load_model("/Users/nikhilanand/FastAPI_BDL_Assignment/training_1/cp.weights.h5")
    digit = await predict_digit(model,flattened_image_list)

    return {"digit":digit}

async def predict_digit(model:Sequential,data_point:list)->str:
    return str(np.argmax(model(np.array(data_point).reshape(1,784))))

async def load_model(path:str) -> Sequential:
    model2 = keras.Sequential()
    model2.add(layers.Dense(256, activation='sigmoid', input_shape=(784,)))
    model2.add(layers.Dense(128, activation='sigmoid'))
    model2.add(layers.Dense(10, activation='softmax'))
    model2.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    model2.load_weights(path)
    return model2