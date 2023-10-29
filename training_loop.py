import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

import networks
from data_processing import CustomImageDataset

import argparse

import time
import csv
import os
import pickle

parser = argparse.ArgumentParser(description='Argument Parser Example')

parser.add_argument('--training', action='store_true', help='Training argument')
parser.add_argument('--name', type=str, help='Name of model')

args = parser.parse_args()

torch.manual_seed(420)

MODEL_DIR = 'models'

# initialize data size
batch_size = 64
img_size = 64
num_channels = 3

img_embedding_size = 1000
text_embedding_size = 768
post_embedding_size = 1000
num_input_texts = 3
num_labels = 423

# device
device = 'cuda' #if torch.cuda.is_available() else 'cpu'

# define hyperparameters
learning_rate = 0.0015
num_epochs = 30

# initialize networks
print('initializing networks')
image_network = networks.StructuralEmbeddingNetwork(img_size, num_channels, img_embedding_size).to(device)
attention_network = networks.TextImageAttention(num_input_texts, text_embedding_size, img_embedding_size, post_embedding_size).to(device)
classifier = networks.PostClassifier(post_embedding_size, num_labels).to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

train_data_path = "data/train.csv"
test_data_path = "data/test.csv"
train_image_text_data_path = "data/train_image_text.csv"
test_image_text_data_path = "data/test_image_text.csv"
train_image_path = "shrunken_images/training"
test_image_path = "shrunken_images/test"


training = args.training
extra = args.name

train_loss_file_path = f"data/train_loss_bs{batch_size}_{extra}.csv"
train_accuracy_file_path = f"data/train_accuracy_bs{batch_size}_{extra}.csv"

test_output_file = f"data/test_output_bs{batch_size}_{extra}.csv"

image_model_save_path = f'{MODEL_DIR}/post_image_model_latest_bs{batch_size}_{extra}.pth'
attention_model_save_path = f'{MODEL_DIR}/post_attention_model_latest_bs{batch_size}_{extra}.pth'
classifier_model_save_path = f'{MODEL_DIR}/post_classifier_model_latest_bs{batch_size}_{extra}.pth'


if training:
    print('setting up for training')
    
    image_network.train()
    attention_network.train()
    classifier.train()
    
    train_dataset_save_path = f'{MODEL_DIR}/train_dataset_save.pkl'
    train_dataset = None
    
    if os.path.exists(train_dataset_save_path) and False:
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
    
    loss_file = open(train_loss_file_path, mode='a', newline='')
    accuracy_file = open(train_accuracy_file_path, mode='a', newline='')
    
    loss_writer = csv.writer(loss_file)
    accuracy_writer = csv.writer(accuracy_file)
    loss_writer.writerow(['Epoch', 'Loss'])
    accuracy_writer.writerow(['Epoch', 'Accuracy'])
    
    if os.path.exists(classifier_model_save_path):
        print('loading model checkpoints')
        try:
            image_state_dict = torch.load(image_model_save_path)
            attention_state_dict = torch.load(attention_model_save_path)
            classifier_state_dict = torch.load(classifier_model_save_path)
            
            image_network.load_state_dict(image_state_dict)
            attention_network.load_state_dict(attention_state_dict)
            classifier.load_state_dict(classifier_state_dict)
            
        except:
            print('one or more pickle files are corrupted, starting from scratch')

    print('beginning training loop')
    for epoch in range(num_epochs):        
        epoch_correct = 0
        epoch_samples = 0
        running_loss = 0.0
        
        for batch_idx, (img, text_embeddings, labels, expected) in enumerate(train_dataloader):
            start_time = time.time()
            
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
            # post_embed = torch.tensor(0)
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
            
            loss_writer.writerow([epoch, loss.item()])
            accuracy_writer.writerow([epoch, acc])
            
            # backpropagation
            loss.backward()
            
            # gradient descent step
            optimizer.step()
            
            end_time = time.time()
            iteration_time = end_time - start_time

            
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%, Time: {iteration_time:.2f} seconds')

        average_loss = running_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}')
        
        
        if os.path.exists(image_model_save_path):
            os.remove(image_model_save_path)
        if os.path.exists(attention_model_save_path):
            os.remove(attention_model_save_path)
        if os.path.exists(classifier_model_save_path):
            os.remove(classifier_model_save_path)

        torch.save(image_network.state_dict(), image_model_save_path)
        torch.save(attention_network.state_dict(), attention_model_save_path)
        torch.save(classifier.state_dict(), classifier_model_save_path)
        
        loss_file.flush()
        accuracy_file.flush()

    
    loss_file.close()
    accuracy_file.close()


print('setting up for testing')

test_dataset_save_path = f'{MODEL_DIR}/test_dataset_save.pkl'
test_dataset = None

if os.path.exists(test_dataset_save_path) and False:
    print(f'Found saved dataset. Loading from {test_dataset_save_path}')
    with open(test_dataset_save_path, 'rb') as file:
        test_dataset = pickle.load(file)
else:
    print('Did not find existing dataset. Initializing...')
    test_dataset = CustomImageDataset(test_data_path, test_image_path, train_image_text_data_path, testing=True)
    with open(test_dataset_save_path, 'wb') as file:
        pickle.dump(test_dataset, file)
        print(f'Loaded data and saved dataset to {test_dataset_save_path}')

test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

if not training:
    image_network.load_state_dict(torch.load(image_model_save_path))
    attention_network.load_state_dict(torch.load(attention_model_save_path))
    classifier.load_state_dict(torch.load(classifier_model_save_path))

image_network.eval()
attention_network.eval()
classifier.eval()

with torch.no_grad(), open(test_output_file, 'w', newline='') as test_out_file:
    test_csv_writer = csv.writer(test_out_file)
    test_csv_writer.writerow(('id', 'approved'))
    for batch_idx, batch in enumerate(test_dataloader):
        # ids = [''.join(id) for id in batch['ids']]
        ids = batch['ids']
        
        img = batch['image'].float().to(device)
        text_embeddings = batch['text_embeddings'].to(device)
        labels = batch['labels'].float().to(device)
        
        # forward pass of networks
        img_embed = image_network(img)
        post_embed = attention_network(img_embed, text_embeddings)
        # post_embed = torch.tensor(0)
        outputs = classifier(post_embed, labels)
        
        for i, output in enumerate(outputs):
            output = "true" if output >= 0.5 else "false"
            test_csv_writer.writerow((ids[i], output))
        
        print(f'Batch {batch_idx}/{len(test_dataloader)}')
        test_out_file.flush()

    print(f'Done testing')
