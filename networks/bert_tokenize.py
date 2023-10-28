import torch
from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self):
        print('initializing BERT tokenizer')
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        
        print('initializing BERT model')
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
    
    def get_embedding(self, input_texts) -> torch.Tensor:
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        
        encoded = torch.tensor([self.tokenizer.encode(*input_texts)])
        text_embeddings = self.model(encoded).pooler_output
        
        return text_embeddings
