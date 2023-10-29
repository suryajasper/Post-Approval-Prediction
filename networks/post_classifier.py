import torch
from torch import nn

class PostClassifier(nn.Module):
    def __init__(self, embedding_size, num_labels):
        super(PostClassifier, self).__init__()
        
        self.embedding_network = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=2000),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=1000),
            nn.ReLU(),
        )

        self.final_ff_network = nn.Sequential(
            nn.Linear(in_features=num_labels + 1000, out_features=2000),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=4000),
            nn.ReLU(),
            nn.Linear(in_features=4000, out_features=8000),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(in_features=8000, out_features=8000),
            nn.ReLU(),
            nn.Linear(in_features=8000, out_features=4000),
            nn.ReLU(),
            nn.Linear(in_features=4000, out_features=2000),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(in_features=2000, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1),
            nn.Sigmoid()
        )
        
    def forward(self, embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embed_proj = self.embedding_network(embedding)
        
        combined_proj = torch.cat((embed_proj, labels), dim=1)
        x = self.final_ff_network(combined_proj)
        
        return x