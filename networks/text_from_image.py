import os
import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from google.cloud import vision
from google.cloud.vision_v1 import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/surya/OneDrive/Documents/tamu-datathon-23-sub-project-fde75f68cf19.json"

client = vision.ImageAnnotatorClient()

def text_from_PNG_image(filepath):

    with open(filepath, 'rb') as image_file:
        content = image_file.read()
    
    image = types.Image(content = content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        return texts[0].description
    return ""