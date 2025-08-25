import torch
import torch.nn as nn

class CnnBlock(nn.Module):
    """ Cnn block for the Siamese Network """

    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=0):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(n_out),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.cnn(x)

class ResidualBlock(nn.Module):
    """ Residual block containing 2 convolutions """

    def __init__(self, n_in, n_out):
        super().__init__()

        self.cnn = nn.Sequential(
            CnnBlock(n_in, n_out, kernel_size=3, stride=1, padding=1),
            CnnBlock(n_in, n_out, kernel_size=3, stride=1, padding=1),
        )


    def forward(self, x):
        x = self.cnn(x) + x
        return x
    
class ZeroPadShortcut(nn.Module):
    """
    Resnet paper option A for downsampling the input's spatial dimensions 
    to match the new outputs padding every other channel with zeros.
    The upside of this is that it adds no extra parameters to the model.
    """
    def __init__(self, input_channels, output_channels, stride=2):
        super(ZeroPadShortcut, self).__init__()
        self.stride = stride
        self.input_channels = input_channels
        self.output_channels = output_channels

    def forward(self, x):
        # Downsample with stride
        x = x[:, :, ::self.stride, ::self.stride]
        # Calculate the padding size to match the output channels
        pad_channels = self.output_channels - self.input_channels
        if pad_channels > 0:
            # Pad zeros to the channel dimension
            padding = torch.zeros(
                (x.size(0), pad_channels, x.size(2), x.size(3)),
                device=x.device,
                dtype=x.dtype
            )
            x = torch.cat([x, padding], dim=1)
        return x
    
class MiniResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.zero_pad1 = ZeroPadShortcut(input_channels=64, output_channels=128)
        self.zero_pad2 = ZeroPadShortcut(input_channels=128, output_channels=256)
        self.zero_pad3 = ZeroPadShortcut(input_channels=256, output_channels=512)

        self.max_pool = nn.MaxPool2d(2)

        #NOTE: ugly but needed to adress self.max_pool(self.cnn3(x)) + res3, 
        # where the first term has spatial dimensions 3x3 and the second 4x4
        self.max_pool3 = nn.Sequential(
                            nn.ZeroPad2d((0, 1, 0, 1)),  # Pad right and bottom by 1
                            nn.MaxPool2d(2)
                        )

        self.first_layer = nn.Sequential(
            # it is recommended, if you use a larger kernel at all, 
            # to only use it at the start. This is due to two reasons:
            # 1. the increase in parameters otherwise would be too large
            # 2. a large kernel can significantly reduce the spatial dimensions (if you use a stride > 1)
            # while still capturing larger contexts in its receptive field, 
            # which makes the subsequent layers run faster
            CnnBlock(1, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(2)
        )

        self.cnn1 = CnnBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.cnn2 = CnnBlock(128, 256, kernel_size=3, stride=1, padding=1)
        self.cnn3 = CnnBlock(256, 512, kernel_size=3, stride=1, padding=1)

        self.res_block1 = ResidualBlock(64, 64)
        self.res_block2 = ResidualBlock(128, 128)
        self.res_block3 = ResidualBlock(256, 256)

        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)  # Global average pooling

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.Sigmoid()
        )
        
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Input shape: (batch_size, 1, 112, 112)

        x = self.first_layer(x) # cnn: (batch_size, 64, 56, 56)
                                # nn.MaxPool2d(2): (batch_size, 64, 28, 28)

        x = self.res_block1(x) # (batch_size, 64, 28, 28)
        res1 = self.zero_pad1(x) # (batch_size, 128, 14, 14)
        x = self.max_pool(self.cnn1(x)) + res1 # cnn: (batch_size, 128, 28, 28)
                                               # maxpool + res1: (batch_size, 128, 14, 14)

        x = self.res_block2(x) # (batch_size, 128, 14, 14)
        res2 = self.zero_pad2(x) # (batch_size, 256, 7, 7)
        x = self.max_pool(self.cnn2(x)) + res2 # cnn: (batch_size, 256, 14, 14)
                                               # maxpool + res2: (batch_size, 256, 7, 7)

        x = self.res_block3(x) # (batch_size, 256, 7, 7)
        # Use adaptive pooling for res3 as well to ensure 4x4 output
        res3 = self.zero_pad3(x) # (batch_size, 512, 4, 4)
        x = self.max_pool3(self.cnn3(x)) + res3 # cnn: (batch_size, 512, 7, 7)
                                                # maxpool + res3: (batch_size, 512, 4, 4)

        x = self.global_avg_pooling(x) # (batch_size, 512, 1, 1)

        x = self.flatten(x) # (batch_size, 512)

        # x = self.fc(x) # linear: (batch_size, 512)
        #                # sigmoid: (batch_size, 512)

        return x