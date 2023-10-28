import os
from PIL import Image

image_path = "images"
training_image_path = "images/training"
test_image_path = "images/test"

shrunken_image_path = "shrunken_images"
shrunken_training_image_path = "shrunken_images/training"
shrunken_test_image_path = "shrunken_images/test"

if not os.path.exists(training_image_path) or not os.path.exists(test_image_path):
    print("You must download the images from kaggle first. Run load_data.py")
    exit()

# Creates image directories if they do not already exist
if not os.path.exists(shrunken_training_image_path) or not os.path.exists(shrunken_test_image_path):
    os.mkdir(shrunken_image_path)
    os.mkdir(shrunken_training_image_path)
    os.mkdir(shrunken_test_image_path)

# Loop through all training images in the directory
for filename in os.listdir(training_image_path):
    read_path = os.path.join(training_image_path, filename)
    write_path = os.path.join(shrunken_training_image_path, filename)
    if os.path.isfile(read_path):
        image = Image.open(read_path)  # Replace with the actual image file path
        image = image.resize((64, 64))
        image.save(write_path)

# Loop through all test images in the directory
for filename in os.listdir(test_image_path):
    read_path = os.path.join(test_image_path, filename)
    write_path = os.path.join(shrunken_test_image_path, filename)
    if os.path.isfile(read_path):
        image = Image.open(read_path)  # Replace with the actual image file path
        image = image.resize((64, 64))
        image.save(write_path)