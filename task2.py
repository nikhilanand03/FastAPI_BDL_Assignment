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

app = FastAPI()

# Root endpoint, returns an empty response
@app.get("/")
async def root():
    return {}

# Endpoint to predict the digit from an uploaded image file
@app.post("/predict")
async def predict(file: UploadFile):
    # Read the content of the uploaded file
    request_object_content = await file.read()
    # Open the image using PIL library
    img = Image.open(io.BytesIO(request_object_content)) 
    
    # Resize and format the image for model prediction
    resized_img = await format_image(img)

    # Convert the image data to a numpy array
    arr = np.array(resized_img)
    
    print(arr,arr.shape)

    # Flatten the image array
    flattened_image=arr.reshape(-1)
    flattened_image_list = flattened_image.tolist()
    
    # Load the trained model
    model = await load_model("/Users/nikhilanand/FastAPI_BDL_Assignment/training_1/cp.weights.h5")
    # Predict the digit using the loaded model
    digit = await predict_digit(model,flattened_image_list)

    # Return the predicted digit
    return {"digit":digit}

# Function to predict the digit using the loaded model
async def predict_digit(model:Sequential,data_point:list)->str:
    return str(np.argmax(model(np.array(data_point).reshape(1,784))))

# Function to load the trained model
async def load_model(path:str) -> Sequential:
    # Define a new Sequential model
    model2 = keras.Sequential()
    # Add layers to the model
    model2.add(layers.Dense(256, activation='sigmoid', input_shape=(784,)))
    model2.add(layers.Dense(128, activation='sigmoid'))
    model2.add(layers.Dense(10, activation='softmax'))
    # Compile the model
    model2.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    # Load the weights from the specified path
    model2.load_weights(path)
    return model2

# Function to format the image for model prediction
async def format_image(img:Image):
    # Convert the image to grayscale
    gray_img = img.convert('L')
    # Resize the image to 28x28 pixels
    resized_img = gray_img.resize((28, 28))
    # Invert the colors of the resized image
    resized_img = PIL.ImageOps.invert(resized_img)
    # Save the resized image (for debugging purposes)
    resized_img.save("resized.jpg")
    return resized_img