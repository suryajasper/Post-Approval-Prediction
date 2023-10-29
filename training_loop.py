import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

import networks
from data_processing import CustomImageDataset

import os
import pickle

torch.manual_seed(69)

MODEL_DIR = 'models'

# initialize data size
batch_size = 16
img_size = 64
num_channels = 3

img_embedding_size = 1000
text_embedding_size = 768
post_embedding_size = 1000
num_input_texts = 3
num_labels = 626

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define hyperparameters
learning_rate = 0.004
num_epochs = 100

# initialize networks
print('initializing networks')
image_network = networks.StructuralEmbeddingNetwork(img_size, num_channels, img_embedding_size).to(device)
attention_network = networks.TextImageAttention(num_input_texts, text_embedding_size, img_embedding_size, post_embedding_size).to(device)
classifier = networks.PostClassifier(post_embedding_size, num_labels).to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

print('initializing dataloader')
train_data_path = "data/small_train.csv"
image_text_data_path = "data/train_image_text.csv"
shrunken_training_image_path = "shrunken_images/training"

dataset_save_path = f'{MODEL_DIR}/dataset_save.pkl'
train_dataset = None

if os.path.exists(dataset_save_path):
    print(f'Found saved dataset. Loading from {dataset_save_path}')
    with open(dataset_save_path, 'rb') as file:
        train_dataset = pickle.load(file)
else:
    print('Did not find existing dataset. Initializing...')
    train_dataset = CustomImageDataset(train_data_path, shrunken_training_image_path, image_text_data_path, testing=False)
    with open(dataset_save_path, 'wb') as file:
        pickle.dump(train_dataset, file)
        print(f'Loaded data and saved dataset to {dataset_save_path}')

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

for epoch in range(num_epochs):
    print(f'Starting epoch {epoch}')
    
    epoch_correct = 0
    epoch_samples = 0
    
    for batch_idx, (img, text_embeddings, labels, expected) in enumerate(train_dataloader):
        img = img.float().to(device)
        text_embeddings = text_embeddings.to(device)
        labels = labels.float().to(device)
        expected = expected.to(device)

        # print('img', img.shape, img.dtype)
        # print('text_embeddings', text_embeddings.shape)
        # print('labels', labels.shape)
        # print('expected', expected.shape)
        
        optimizer.zero_grad()
        
        # forward pass of networks
        img_embed = image_network(img)
        post_embed = attention_network(img_embed, text_embeddings)
        outputs = classifier(post_embed, labels)
        
        outputs = outputs.squeeze()
        
        # compute loss
        loss = criterion(outputs, expected.float())
        
        # update epoch accuracy 
        predicted = (outputs >= 0.5).float()
        epoch_correct += (predicted == expected).sum().item()
        epoch_samples += len(expected)
        acc = epoch_correct / epoch_samples * 100
        
        # backpropagation
        loss.backward()
        
        # gradient descent step
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%')

    torch.save(image_network.state_dict(), f'{MODEL_DIR}/post_image_model_latest.pth')
    torch.save(attention_network.state_dict(), f'{MODEL_DIR}/post_attention_model_latest.pth')
    torch.save(classifier.state_dict(), f'{MODEL_DIR}/post_classifier_model_latest.pth')