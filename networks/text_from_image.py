import os
from google.cloud import vision
from google.cloud.vision_v1 import types
import io
from PIL import Image

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/ash/Documents/GitHub_Projects/tamu-datathon-23-sub-project-fde75f68cf19.json"

client = vision.ImageAnnotatorClient()

def text_from_PIL_image(filename):

    img_byte_array = io.BytesIO()
    image_data = img_byte_array.getvalue()

    image = types.Image(content=image_data)
    response = client.text_detection(image=image)

    texts = response.text_annotations
    return(texts[0].description)