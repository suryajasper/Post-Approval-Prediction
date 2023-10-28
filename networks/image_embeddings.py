import torch
from torch import nn

# VGG16 implementation
class VGG16EmbeddingNetwork(nn.Module):
    def __init__(self, 
                 img_size=256,
                 num_channels=3,
                 device='cpu',
                 precision='fp32',
                 num_conv_layers=5):
        
        super.__init__(self)
        
        # ensure we can safely downscale our image
        assert(img_size % pow(2, num_conv_layers-1) == 0), \
            f'Image size {img_size} cannot be downscaled through {num_conv_layers} convolutions'
        
        self.img_size = img_size
        self.num_channels = num_channels
        
        self.use_fp16 = precision != 'fp32'
        self.device = device
        
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
                 batch_size=16,
                 img_size=64,
                 num_channels=3,
                 embedding_size=1000,
                 device='cuda'):
        
        super.__init__(self)
        
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.img_size = img_size
        self.embedding_size = embedding_size
        self.device = device
        
        size_log2 = torch.log2(img_size)
        assert torch.ceil(size_log2) == torch.floor(size_log2), \
            'image size must be a power of 2'
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax2d()
        
        conv_stride = 2
        
        self.conv_layers = [
            [
                nn.Conv2d(in_channels=num_channels, out_channels=32, stride=conv_stride, device=self.device),
                nn.Conv2d(in_channels=32, out_channels=32, stride=conv_stride, device=self.device),
            ],
            [
                nn.Conv2d(in_channels=32, out_channels=64, stride=conv_stride, device=self.device),
                nn.Conv2d(in_channels=64, out_channels=64, stride=conv_stride, device=self.device),
            ],
            [
                nn.Conv2d(in_channels=64, out_channels=128, stride=conv_stride, device=self.device),
                nn.Conv2d(in_channels=128, out_channels=128, stride=conv_stride, device=self.device),
                nn.Conv2d(in_channels=128, out_channels=128, stride=conv_stride, device=self.device),
            ],
        ]
        
        self.dense = nn.Sequential(
            nn.Linear(in_features=batch_size*4096, out_features=batch_size*4096, device=self.device),
            self.relu,
            nn.Dropout2d(p=0.5),
        )
        
        self.linear_start = nn.Linear(in_features=batch_size*32768, out_features=batch_size*4096, device=self.device)
        self.linear_end = nn.Linear(in_features=batch_size*4096, out_features=batch_size*embedding_size, device=self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) > 1:
            x = x.flatten()
        
        # scale down image to 64x64
        size = self.img_size
        
        while size > 64:
            x = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, stride=1)
            x = self.relu(x)
            x = self.maxpool(x)
            size /= 2
        
        assert size == 64, 'downscale failure in forward pass'
        
        # convolutional layers + maxpooling
        for layer_convs in self.conv_layers:
            for conv2d in layer_convs:
                x = conv2d(x)
                x = self.relu(x)
            x = self.maxpool(x)
        
        # fully connected linear layers
        x = self.linear_start(x)
        x = self.relu(x)
        x = self.dense(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.linear_end(x)
        
        # normalize & reshape to output shape (batch_size, embedding_size)
        x = self.softmax(x)
        x = x.reshape((self.batch_size, self.embedding_size))
