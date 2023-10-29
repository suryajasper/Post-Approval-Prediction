import os
import pandas as pd
import requests
#from networks import text_from_PNG_image
#import load_data
import sys
current_directory = os.getcwd()
# parent_directory = os.path.dirname(current_directory)
sys.path.append(current_directory+"/networks")
from text_from_image import text_from_PNG_image

# Basically calling load data
image_train_path = "./data_processing/social-media-post-approval-prediction-with-marky/trainimage.csv"
image_test_path = "./data_processing/social-media-post-approval-prediction-with-marky/testimage.csv"


train_data_path = "./data_processing/social-media-post-approval-prediction-with-marky/train.csv"
test_data_path = "./data_processing/social-media-post-approval-prediction-with-marky/test.csv"
image_path = "data_processing/images"
training_image_path = "data_processing/images/training"
test_image_path = "data_processing/images/test"

# Dataframes
train = pd.read_csv(train_data_path)
test = pd.read_csv(test_data_path)

trainList = []
train_new = train.iloc[:, [0]].copy()
    
trainList = ['' for _ in range(train_new.shape[0])]

for index, row in train.iterrows():
    image = f"{training_image_path}/image_" + row['id'] + ".png"
    trainList[index]= (text_from_PNG_image(image))
    if index % 100 == 0:
        train_new['text_f_image'] = trainList
        train_new.to_csv(image_test_path)

train_new.to_csv(image_train_path)

testList = []
test_new = test.iloc[:, [0]].copy()
testList = ['' for _ in range(test_new.shape[0])]
    
for index, row in test.iterrows():
    image = f"{test_image_path}/image_" + row['id'] + ".png"
    testList[index]= (text_from_PNG_image(image))
    #testList.append(text_from_PNG_image(image))
test_new['text_f_image'] = testList

test_new.to_csv(image_test_path)
