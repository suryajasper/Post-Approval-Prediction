import os
import pandas as pd
import requests

train_data_path = "./social-media-post-approval-prediction-with-marky/small_train.csv"
image_path = "images"
training_image_path = "images/small_training"

def download_image(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to download image from {url}")

if not os.path.isfile(train_data_path):
    print("CSV data must be downloaded")
else:
    print("Marky CSV detected.")

# Dataframes
train = pd.read_csv(train_data_path)

# Creates image directories if they do not already exist
if not os.path.exists(training_image_path):
    os.mkdir(training_image_path)

# Download all the training images
for index, row in train.iterrows():
    post_id = row['id'] 
    image_url = row['image'] 
    image_filename = os.path.join(training_image_path, f'image_{post_id}.png')
    # Checking if image is downloaded
    if (not os.path.isfile(image_filename)):
        download_image(image_url, image_filename)