import os
import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from google.cloud import vision
from google.cloud.vision_v1 import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/ash/Documents/GitHub_Projects/tamu-datathon-23-sub-project-fde75f68cf19.json"

client = vision.ImageAnnotatorClient()

def text_from_torch_tensor():

    tensor_to_PIL = transforms.ToPILImage()
    pil_image = tensor_to_PIL

    img_byte_array = io.BytesIO()
    pil_image.save(img_byte_array, format='PNG')
    image_data = img_byte_array.getvalue()

    image = types.Image(content=image_data)
    response = client.text_detection(image=image)

    texts = response.text_annotations
    if texts:
        return texts[0].description