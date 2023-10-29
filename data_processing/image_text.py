import os
import pandas as pd
import requests
#from networks import text_from_PNG_image
#import load_data
import sys
from tqdm import tqdm
current_directory = os.getcwd()
# parent_directory = os.path.dirname(current_directory)
sys.path.append(current_directory+"/networks")
from text_from_image import text_from_PNG_image

# Basically calling load data
image_train_path = "data/train_image_text.csv"
image_test_path = "data/test_image_text.csv"

train_data_path = "data/train.csv"
test_data_path = "data/test.csv"
image_path = "images"
training_image_path = "images/training"
test_image_path = "images/test"

# Dataframes
train = pd.read_csv(train_data_path)
test = pd.read_csv(test_data_path)

# trainList = []
# train_new = train.iloc[:, [0]].copy()
    
# trainList = ['' for _ in range(train_new.shape[0])]

# training_bar = tqdm(total=len(trainList), desc="Extracting training text", unit="image")
# for index, row in train.iterrows():
#     image = f"{training_image_path}/image_" + row['id'] + ".png"
#     trainList[index]= (text_from_PNG_image(image))
#     if index % 100 == 0:
#         train_new['text_f_image'] = trainList
#         train_new.to_csv(image_train_path)
#     training_bar.update(1)

# train_new['text_f_image'] = trainList
# train_new.to_csv(image_train_path)

testList = []
test_new = test.iloc[:, [0]].copy()
testList = ['' for _ in range(test_new.shape[0])]

testing_bar = tqdm(total=len(testList), desc="Extracting testing text", unit="image")
for index, row in test.iterrows():
    image = f"{test_image_path}/image_" + row['id'] + ".png"
    testList[index]= (text_from_PNG_image(image))
    testing_bar.update(1)
    
test_new['text_f_image'] = testList
test_new.to_csv(image_test_path)
