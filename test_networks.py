import torch
from networks import *

def test_image_embedding(batch_size=16, image_size=64):
    print('--TESTING image embedding network')

    img = torch.randn((batch_size, 3, image_size, image_size), device='cuda')
    
    model = StructuralEmbeddingNetwork().to('cuda')
    embed = model(img)
    
    print(f'--SUCCESS: input {img.shape} -> embedding {embed.shape}')

def test_attention(batch_size=16, text_embed_size=756, image_embed_size=1000, hidden_size=500):
    print('--TESTING self-attention network')

    img_embedding = torch.randn((batch_size, image_embed_size), device='cuda')
    text_embeddings = torch.randn((batch_size, 3, text_embed_size), device='cuda')
    
    model = TextImageAttention(3, text_embed_size, image_embed_size, hidden_size).to('cuda')
    post_embedding = model(img_embedding, text_embeddings)
    
    print(f'--SUCCESS: img embed {img_embedding.shape} + text embed {text_embeddings.shape} -> post embed {post_embedding.shape}')

def test_classification(embedding_size=500, num_labels=250, batch_size=16):
    print('--TESTING classification network')

    embedding = torch.randn((batch_size, embedding_size), device='cuda')
    one_hot_labels = torch.randint(0, 2, (batch_size, num_labels), device='cuda').float()
    
    model = PostClassifier(embedding_size, num_labels).to('cuda')
    classification = model(embedding, one_hot_labels)
    
    print(f'--SUCCESS: embedding {embedding.shape} + labels {one_hot_labels.shape} -> classification {classification.shape}')

if __name__ == '__main__':
    test_image_embedding()
    test_attention()
    test_classification()
    