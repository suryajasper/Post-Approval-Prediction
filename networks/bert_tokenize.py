import torch

class TextTokenizer:
    def __init__(self):
        print('initializing BERT tokenizer')
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        
        print('initializing BERT model')
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
    
    def get_embedding(self, input_texts) -> torch.Tensor:
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        
        try:
            input_texts = [text[:500] for text in input_texts]
        except:
            print('text trimming failed -> trying encoder')
        
        try:
            encoded = torch.tensor([self.tokenizer.encode(*input_texts)])
            text_embeddings = self.model(encoded).last_hidden_state
            text_embeddings = torch.mean(text_embeddings.squeeze(), dim=0)
            return text_embeddings
        except:
            print('text encoding failed -> returning zero vector')
            return torch.zeros((768))
