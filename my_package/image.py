
from PIL import Image
from io import BytesIO
import logging
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO)
async def process_image(image):
    logging.info('Start image processing')
    contents = await image.read()
    image = Image.open(BytesIO(contents))

    logging.info(image)
    image_info = {
        'width': image.width,
        'height': image.height,
        'file': image.format,
    }

    return {
        "filename": 'temp',
        "format": image.format,
        "size": image.size,
        "mode": image.mode
    }

def validate_image_file(filetype):

    if filetype not in ["image/jpeg", "image/png", "image/gif"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
