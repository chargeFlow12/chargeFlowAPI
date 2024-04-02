from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile,File
import logging
from fastapi.middleware.cors import CORSMiddleware

from my_package.image import process_image,validate_image_file,sample_process_image

app = FastAPI()

origin=['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origin,
    allow_methods=["GET","POST"],
    allow_headers=["*"])

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post('/image')
async def upload_image(image: UploadFile= File(...)):
    validate_image_file(image.content_type)
    image_info = await sample_process_image(image)
    return image_info

