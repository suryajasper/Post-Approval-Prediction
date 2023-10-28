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
