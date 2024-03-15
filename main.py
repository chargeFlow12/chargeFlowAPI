from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile,File
import logging

from my_package.image import process_image,validate_image_file

app = FastAPI()



@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post('/image')
async def upload_image(image: UploadFile= File(...)):
    validate_image_file(image.content_type)
    image_info = await process_image(image)
    return image_info

