import torch
from torch import nn

class TextImageAttention(nn.Module):
    def __init__(self,
                 num_input_texts: int,
                 text_embed_size: int,
                 image_embed_size: int,
                 hidden_size: int):

        super(TextImageAttention, self).__init__()
        
        if hidden_size is None:
            hidden_size = min(image_embed_size, text_embed_size)
        
        # define linear layers
        self.image_layer = nn.Linear(in_features=image_embed_size, out_features=hidden_size)
        
        self.text_layers = nn.ModuleList([
            nn.Linear(in_features=text_embed_size, out_features=hidden_size)
                for _ in range(num_input_texts)
        ])
        
        self.merge_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size*(num_input_texts+1), out_features=hidden_size*num_input_texts),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size*num_input_texts, out_features=hidden_size),
            nn.ReLU(),
        )
        
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        
        # define attention parameters
        self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=True)
        self.b = nn.Parameter(torch.Tensor(hidden_size), requires_grad=True)
    
    def forward(self, image_embedding, text_embeddings) -> torch.Tensor:
        l_combined = [self.image_layer(image_embedding)]
        
        for i, text_layer in enumerate(self.text_layers):
            embed = text_layer(text_embeddings[:, i])
            l_combined.append(embed)
        
        combined_proj = torch.cat(l_combined, dim=1)
        combined_proj = self.merge_layer(combined_proj)

        attention_scores = torch.matmul(combined_proj, self.W) + self.b
        attention_weights = nn.functional.softmax(attention_scores, dim=1)

        output = attention_weights * combined_proj
        
        output = self.output_layer(output)

        return output