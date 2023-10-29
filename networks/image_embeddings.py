import torch
from torch import nn

# VGG16 implementation
class VGG16EmbeddingNetwork(nn.Module):
    def __init__(self, 
                 img_size=256,
                 num_channels=3,
               
                 precision='fp32',
                 num_conv_layers=5):
        
        super(VGG16EmbeddingNetwork, self).__init__()
        
        # ensure we can safely downscale our image
        assert(img_size % pow(2, num_conv_layers-1) == 0), \
            f'Image size {img_size} cannot be downscaled through {num_conv_layers} convolutions'
        
        self.img_size = img_size
        self.num_channels = num_channels
        
        self.use_fp16 = precision != 'fp32'
        
        self.conv2ds = self.build_conv_layers(num_conv_layers)
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        
    def build_conv_layers(self, num_layers):
        convs = []
        
        channels = self.num_channels
        
        for i in range(num_layers):
            new_channels = channels * 2 if i < num_layers-1 else channels
            
            layer_convs = []
            
            # add convolutional layers
            layer_convs.append(
                nn.Conv2d(in_channels=channels, out_channels=new_channels, kernel_size=3, stride=1)
            )
            
            sub_layers = 2 if i < num_layers / 2 else 3
            layer_convs.extend([
                nn.Conv2d(in_channels=new_channels, out_channels=new_channels) 
                    for _ in range(sub_layers)
            ])
            
            channels = new_channels
            
            convs.append(layer_convs)
        
        return convs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer_convs in enumerate(self.conv2ds):
            for conv2d in layer_convs:
                x = conv2d(x)
                x = self.relu(x)
            x = self.maxpool2d(x)
        
        x = x.flatten()

class StructuralEmbeddingNetwork(nn.Module):
    def __init__(self,
                 img_size=64,
                 num_channels=3,
                 embedding_size=1000):
        
        super(StructuralEmbeddingNetwork, self).__init__()
        
        self.num_channels = num_channels
        self.img_size = img_size
        self.embedding_size = embedding_size
        
        size_log2 = torch.log2(torch.tensor(img_size))
        assert torch.ceil(size_log2) == torch.floor(size_log2), \
            'image size must be a power of 2'
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax2d()
        
        self.downscale_conv = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
        
        self.conv_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            ]), 
            nn.ModuleList([
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            ]), 
            nn.ModuleList([
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            ]),
        ])
        
        self.dense = nn.Sequential(
            nn.Linear(in_features=2048, out_features=2048),
            self.relu,
            # nn.Dropout2d(p=0.5),
        )
        
        self.linear_start = nn.Linear(in_features=4096, out_features=2048)
        self.linear_end = nn.Linear(in_features=2048, out_features=embedding_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        # scale down image to 64x64
        size = self.img_size
        
        while size > 64:
            x = self.downscale_conv(x)
            x = self.relu(x)
            x = self.maxpool(x)
            size /= 2
        
        assert size == 64, 'downscale failure in forward pass'
        
        # convolutional layers + maxpooling
        for conv_group in self.conv_layers:
            for conv_layer in conv_group:
                x = conv_layer(x)
                x = self.relu(x)
            x = self.maxpool(x)
        
        x = x.view(x.shape[0], -1)
        
        # fully connected linear layers
        x = self.linear_start(x)
        x = self.relu(x)
        # x = self.dense(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.linear_end(x)
        # x = self.softmax(x)
        
        return x