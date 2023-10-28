import torch
from torch import nn

class TextImageAttention(nn.Module):
    def __init__(self,
                 text_embed_sizes: list(int),
                 image_embed_size: int,
                 hidden_size: int):

        super(TextImageAttention, self).__init__()
        
        if hidden_size is None:
            hidden_size = min(image_embed_size, *text_embed_sizes)
        
        # define linear layers
        self.image_layer = nn.Linear(in_features=image_embed_size, out_features=hidden_size)
        
        self.text_layers = [
            nn.Linear(in_features=embed_size, out_features=hidden_size)
                for embed_size in text_embed_sizes
        ]
        
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        
        # define attention parameters
        self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=True)
        self.b = nn.Parameter(torch.Tensor(hidden_size), requires_grad=True)
    
    def forward(self, image_embedding, text_embeddings) -> torch.Tensor:
        l_combined = [self.image_layer(image_embedding)]
        
        for i, text_layer in enumerate(self.text_layers):
            l_combined.append(text_layer(text_embeddings[i]))
        
        combined_proj = torch.cat(l_combined, dim=1)

        attention_scores = torch.matmul(combined_proj, self.W) + self.b
        attention_weights = nn.functional.softmax(attention_scores, dim=1)

        output = torch.sum(attention_weights * combined_proj, dim=1)
        output = self.output_layer(output)

        return output