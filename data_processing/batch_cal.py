import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        id = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        image = read_image(img_path)
        # Columns 7, 11, 
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
data = CustomImageDataset("./social-media-post-approval-prediction-with-marky/train.csv", "shrunken_images/training")
batchsize = 32
shuffle_bool = True

testdata = DataLoader(data, 32, True)
print(testdata)

