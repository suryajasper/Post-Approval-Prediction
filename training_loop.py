import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

import networks
from data_processing import CustomImageDataset

import csv
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
test_data_path = "data/test.csv"
train_image_text_data_path = "data/train_image_text.csv"
test_image_text_data_path = "data/test_image_text.csv"
train_image_path = "shrunken_images/training"
test_image_path = "shrunken_images/test"

loss_file_path = "data/train_loss.csv"
accuracy_file_path = "data/train_accuracy.csv"

image_model_save_path = f'{MODEL_DIR}/post_image_model_latest.pth'
attention_model_save_path = f'{MODEL_DIR}/post_attention_model_latest.pth'
classifier_model_save_path = f'{MODEL_DIR}/post_classifier_model_latest.pth'

training = False

if training:
    print('setting up for training')
    
    train_dataset_save_path = f'{MODEL_DIR}/train_dataset_save.pkl'
    train_dataset = None
    
    if os.path.exists(train_dataset_save_path):
        print(f'Found saved dataset. Loading from {train_dataset_save_path}')
        with open(train_dataset_save_path, 'rb') as file:
            train_dataset = pickle.load(file)
    else:
        print('Did not find existing dataset. Initializing...')
        train_dataset = CustomImageDataset(train_data_path, train_image_path, train_image_text_data_path, testing=False)
        with open(train_dataset_save_path, 'wb') as file:
            pickle.dump(train_dataset, file)
            print(f'Loaded data and saved dataset to {train_dataset_save_path}')
    
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch}')
        
        epoch_correct = 0
        epoch_samples = 0
        running_loss = 0.0
        
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
            
            running_loss += loss.item()
            train_losses.append(loss.item())
            train_accuracies.append(acc)
            
            # backpropagation
            loss.backward()
            
            # gradient descent step
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%')

        average_loss = running_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}')

        torch.save(image_network.state_dict(), image_model_save_path)
        torch.save(attention_network.state_dict(), attention_model_save_path)
        torch.save(classifier.state_dict(), classifier_model_save_path)

else:
    print('setting up for testing')
    
    test_dataset_save_path = f'{MODEL_DIR}/test_dataset_save.pkl'
    test_dataset = None
    
    if os.path.exists(test_dataset_save_path):
        print(f'Found saved dataset. Loading from {test_dataset_save_path}')
        with open(test_dataset_save_path, 'rb') as file:
            test_dataset = pickle.load(file)
    else:
        print('Did not find existing dataset. Initializing...')
        test_dataset = CustomImageDataset(test_data_path, test_image_path, train_image_text_data_path, testing=True)
        with open(test_dataset_save_path, 'wb') as file:
            pickle.dump(test_dataset, file)
            print(f'Loaded data and saved dataset to {test_dataset_save_path}')

    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
    
    image_network.load_state_dict(torch.load(image_model_save_path))
    attention_network.load_state_dict(torch.load(attention_model_save_path))
    classifier.load_state_dict(torch.load(classifier_model_save_path))
    
    image_network.eval()
    attention_network.eval()
    classifier.eval()
    
    test_losses = []
    test_accuracies = []
    
    test_correct = 0
    test_samples = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (img, text_embeddings, labels) in enumerate(test_dataloader):
            print('shapes', img.shape, text_embeddings.shape, labels.shape)
            img = img.float().to(device)
            text_embeddings = text_embeddings.to(device)
            labels = labels.float().to(device)
            
            # forward pass of networks
            img_embed = image_network(img)
            post_embed = attention_network(img_embed, text_embeddings)
            outputs = classifier(post_embed, labels)

            # compute loss
            loss = criterion(outputs, labels)

            # compute accuracy
            predicted = (outputs > 0.5).float()
            test_correct += (predicted == labels).sum().item()
            test_samples += labels.size(0)
            acc = (test_correct / test_samples) * 100  # Compute accuracy as a percentage

            # store test loss and accuracy values
            test_losses.append(loss.item())
            test_accuracies.append(acc)
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(test_dataloader)}, Loss: {(sum(test_losses)/len(test_losses)):.4f}, Accuracy: {(sum(test_accuracies)/len(test_accuracies)):.2f}%')

        average_test_loss = sum(test_losses) / len(test_losses)
        average_test_acc = sum(test_accuracies) / len(test_accuracies)

        print(f'Test Results - Loss: {average_test_loss:.4f}, Accuracy: {average_test_acc:.2f}%')
