import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

import networks

# initialize data size
img_size = 64
num_channels = 3

img_embedding_size = 1000
text_embedding_size = 656
post_embedding_size = 1000
num_input_texts = 3
num_labels = 10

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define hyperparameters
learning_rate = 0.001
num_epochs = 10

# initialize networks
print('initializing networks')
image_network = networks.StructuralEmbeddingNetwork(img_size, num_channels, img_embedding_size).to(device)
attention_network = networks.TextImageAttention(num_input_texts, text_embedding_size, img_embedding_size, post_embedding_size).to(device)
classifier = networks.PostClassifier(post_embedding_size, num_labels).to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

print('initializing dataloader')
data_loader = DataLoader()

# Assuming you have a DataLoader object called 'data_loader'
for epoch in range(num_epochs):
    print(f'Starting epoch {epoch}')
    
    for batch_idx, (img, text_embeddings, labels, expected) in enumerate(data_loader):
        optimizer.zero_grad()
        
        # batch_size * nc * size * size -> batch_size * size * size * nc
        img = img.permute((0, 2, 3, 1)).to(device)
        
        img_embed = image_network(img)
        post_embed = attention_network(img_embed, text_embeddings)
        outputs = classifier(post_embed, labels.float())
        
        loss = criterion(outputs, expected.float())
        
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}')

    # Save the model at the end of each epoch
    torch.save(image_network.state_dict(), f'post_classifier_model_epoch_{epoch + 1}.pth')
    torch.save(attention_network.state_dict(), f'post_classifier_model_epoch_{epoch + 1}.pth')
    torch.save(classifier.state_dict(), f'post_classifier_model_epoch_{epoch + 1}.pth')