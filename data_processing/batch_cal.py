import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.io import read_image
import sys
import one_hot_encoder

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory+"/networks")
from networks import TextTokenizer

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, testing=False, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.tokenizer = TextTokenizer()
        self.label_tensor = one_hot_encoder.one_hot_encode(self.img_labels)

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

        text_embed_size = 768

        text_embeddings = torch.zeros((2, text_embed_size))
        text_embeddings[0] = self.tokenizer.get_embedding(caption)
        text_embeddings[1] = self.tokenizer.get_embedding(title)
        
        all_labels.append(tone)
        all_labels.append(switchboard_template)
        all_labels.append(theme)
        all_labels.append(prompt_template)
        all_labels.append(photo_template)
        all_labels.append(has_logo)

        image = read_image(img_path)[:3, ...]
        # Columns 7, 11, 
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        #return image, id, caption, title, summary, tone, switchboard_template, theme, prompt_template, photo_template, has_logo
        if(not self.testing):
            return image, text_embeddings, self.label_tensor[idx], self.img_labels.iloc[idx, 15]
        else:
            return image, text_embeddings, self.label_tensor[idx]

train_data_path = "./social-media-post-approval-prediction-with-marky/small_train.csv"
shrunken_training_image_path = "shrunken_images/training"

data = CustomImageDataset(train_data_path, shrunken_training_image_path)
batchsize = 32
shuffle_bool = True

testdata = DataLoader(data, 32, True)