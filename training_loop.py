import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

import networks

# initialize data size
img_size = 64
num_channels = 3

img_embedding_size = 1000
text_embedding_sizes = (500, 500, 500)
post_embedding_size = 1000
num_labels = 10

# define hyperparameters
learning_rate = 0.001
num_epochs = 10

# initialize networks
image_network = networks.StructuralEmbeddingNetwork(img_size=img_size, num_channels=num_channels, embedding_size=img_embedding_size)
attention_network = networks.TextImageAttention(text_embedding_sizes, img_embedding_size, post_embedding_size)
classifier = networks.PostClassifier(post_embedding_size, num_labels)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

data_loader = DataLoader()

# Assuming you have a DataLoader object called 'data_loader'
for epoch in range(num_epochs):
    for batch_idx, (img, text_embeddings, labels, expected) in enumerate(data_loader):
        optimizer.zero_grad()
        
        img_embed = image_network(img)
        post_embed = attention_network(img_embed, text_embeddings)
        outputs = classifier(post_embed, labels)
        
        loss = criterion(outputs, expected.float())
        
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}')

    # Save the model at the end of each epoch
    torch.save(image_network.state_dict(), f'post_classifier_model_epoch_{epoch + 1}.pth')
    torch.save(attention_network.state_dict(), f'post_classifier_model_epoch_{epoch + 1}.pth')
    torch.save(classifier.state_dict(), f'post_classifier_model_epoch_{epoch + 1}.pth')