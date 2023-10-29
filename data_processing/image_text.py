import os
import pandas as pd
import requests
from tqdm import tqdm
from networks import text_from_image
import load_data

# Basically calling load data
image_train_path = "./social-media-post-approval-prediction-with-marky/trainimage.csv"
image_test_path = "./social-media-post-approval-prediction-with-marky/testimage.csv"


train_data_path = "./social-media-post-approval-prediction-with-marky/train.csv"
test_data_path = "./social-media-post-approval-prediction-with-marky/test.csv"
image_path = "images"
training_image_path = "images/training"
test_image_path = "images/test"

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
test = pd.read_csv(test_data_path)

# Creates image directories if they do not already exist
if not os.path.exists(training_image_path) or not os.path.exists(test_image_path):
    os.mkdir(image_path)
    os.mkdir(training_image_path)
    os.mkdir(test_image_path)

training_bar = tqdm(total=train.shape[0], desc="Downloading training images", unit="image")
# Download all the training images
for index, row in train.iterrows():
    post_id = row['id'] 
    image_url = row['image'] 
    image_filename = os.path.join(training_image_path, f'image_{post_id}.png')
    # Checking if image is downloaded
    if (not os.path.isfile(image_filename)):
        download_image(image_url, image_filename)
    training_bar.update(1)

testing_bar = tqdm(total=train.shape[0], desc="Downloading testing images", unit="image")
# Download all the testing images
for index, row in test.iterrows():
    post_id = row['id'] 
    image_url = row['image'] 
    image_filename = os.path.join(test_image_path, f'image_{post_id}.png')
    # Downloads image if it is not already downloaded
    if (not os.path.isfile(image_filename)):
        download_image(image_url, image_filename)
    testing_bar.update(1)

trainList = []
train_new = train.iloc[:, [0]].copy()
    
for index, row in test.iterrows():
    image = "image_" + row['id'] + ".png"
    trainList.append(text_from_image.text_from_PNG_image(image))
train_new['text_f_image'] = trainList

train_new.to_csv(image_train_path)

testList = []
test_new = test.iloc[:, [0]].copy()

    
for index, row in test.iterrows():
    image = "image_" + row['id'] + ".png"
    testList.append(text_from_image.text_from_PNG_image(image))
test_new['text_f_image'] = testList

test_new.to_csv(image_test_path)
