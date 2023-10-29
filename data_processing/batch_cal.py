import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.io import read_image
import sys
from data_processing import one_hot_encoder

# current_directory = os.getcwd()
# parent_directory = os.path.dirname(current_directory)
# sys.path.append(current_directory+"/networks")
from networks import TextTokenizer

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, text_in_images_file, testing=False, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.text_in_images = pd.read_csv(text_in_images_file)
        self.transform = transform
        self.target_transform = target_transform
        self.tokenizer = TextTokenizer()
        self.label_tensor = one_hot_encoder.one_hot_encode(self.img_labels)
        self.testing = testing
        if not self.testing:
            self.balance_labels()
    
    def balance_labels(self):
        approved_posts = self.img_labels[self.img_labels['approved'] == True]
        unapproved_posts = self.img_labels[self.img_labels['approved'] == False]
        print(f'balancing post data : {len(approved_posts)} approved, {len(unapproved_posts)} unapproved')

        min_count = min(approved_posts.shape[0], unapproved_posts.shape[0])
        
        balanced_df = pd.concat([approved_posts.sample(min_count), unapproved_posts.sample(min_count)])
        balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
        
        new_num_approved = len(balanced_df[balanced_df['approved'] == True])
        new_num_unapproved = len(balanced_df[balanced_df['approved'] == False])
        
        assert new_num_approved == new_num_unapproved, 'balancing data failed'
        
        print(f'balanced post data : {new_num_approved} + {new_num_unapproved}')
        self.img_labels = balanced_df

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # print(idx, self.img_labels['id'][idx])        
        id = self.img_labels['id'][idx]
        caption = self.img_labels['caption'][idx]
        title = self.img_labels['parameters_chapter_title'][idx]
        img_path = self.img_dir+"/image_" + self.img_labels.iloc[idx, 0] + ".png"#os.path.join(self.img_dir, "image_" + self.img_labels.iloc[idx, 0] + ".png")

        text_from_image = self.text_in_images.iloc[idx, 2]

        text_embed_size = 768

        text_embeddings = torch.zeros((3, text_embed_size))
        text_embeddings[0] = self.tokenizer.get_embedding(caption)
        text_embeddings[1] = self.tokenizer.get_embedding(title)
        if isinstance(text_from_image, str):
            text_embeddings[2] = self.tokenizer.get_embedding(text_from_image)
        else:
            text_embeddings[2] = torch.zeros((768))

        image = read_image(img_path)[:3, ...]
        if self.transform:
            image = self.transform(image)
        
        if(not self.testing):
            return image, text_embeddings, self.label_tensor[idx], int(self.img_labels.iloc[idx, 15])
        else:
            return {"ids": id, "image": image, "text_embeddings": text_embeddings, "labels": self.label_tensor[idx]}
