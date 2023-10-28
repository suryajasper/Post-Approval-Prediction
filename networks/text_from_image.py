import os
from google.cloud import vision
from google.cloud.vision_v1 import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/ash/Documents/GitHub_Projects/tamu-datathon-23-sub-project-fde75f68cf19.json"

client = vision.ImageAnnotatorClient()

with open("/Users/ash/Desktop/umm.png", 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)
response = client.text_detection(image=image)

texts = response.text_annotations
print(texts[0].description)