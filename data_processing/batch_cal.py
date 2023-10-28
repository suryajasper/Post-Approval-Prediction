import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.io import read_image
import sys

# Get the current directory
current_directory = os.getcwd()

# Get the path to the parent (higher-level) directory
parent_directory = os.path.dirname(current_directory)

# Add the parent directory to sys.path to enable importing
sys.path.append(parent_directory)

# Now you can import the Python script located in the higher-level folder
import bert_tokenize.py

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        all_labels = []
        
        id = self.img_labels.iloc[idx, 0]
        caption = self.img_labels.iloc[idx, 4]
        title = self.img_labels.iloc[idx, 5]
        summary = self.img_labels.iloc[idx, 6]
        tone = self.img_labels.iloc[idx, 7]
        switchboard_template = self.img_labels.iloc[idx, 8]
        theme = self.img_labels.iloc[idx, 9]
        prompt_template = self.img_labels.iloc[idx, 10]
        photo_template = self.img_labels.iloc[idx, 11]
        has_logo = self.img_labels.iloc[idx, 12]
        img_path = self.img_dir+"/image_" + self.img_labels.iloc[idx, 0] + ".png"#os.path.join(self.img_dir, "image_" + self.img_labels.iloc[idx, 0] + ".png")

        all_labels.append(id)
        all_labels.append(caption)
        all_labels.append(title)
        all_labels.append(summary)
        all_labels.append(tone)
        all_labels.append(switchboard_template)
        all_labels.append(theme)
        all_labels.append(prompt_template)
        all_labels.append(photo_template)
        all_labels.append(has_logo)

        image = read_image(img_path)
        # Columns 7, 11, 
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        #return image, id, caption, title, summary, tone, switchboard_template, theme, prompt_template, photo_template, has_logo
        return image, all_labels

train_data_path = "./social-media-post-approval-prediction-with-marky/small_train.csv"
shrunken_training_image_path = "shrunken_images/training"

data = CustomImageDataset(train_data_path, shrunken_training_image_path)
batchsize = 32
shuffle_bool = True

testdata = DataLoader(data, 32, True)


'''
train_features, train_labels = next(iter(testdata))

print(train_labels)

print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
print(f"Label: {label}")
'''