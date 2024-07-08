# There are 8 layers. 5 convolutional and 3 fully connected
# 3rd and 4th layers don't have pooling and normalization and some have it.
# Basically, the first 5 layers do a lot of the image work
# The last 3 layers are related to droputs, organizing and reducing computing time.


# CNN Layers
# 1st layer -> dim = 224 x 224 x 3, output dim -> 11 x 11 x 3, kernels = 96, stride = 4
# 2nd layer -> dim = 11 x 11 x 3, output dim -> 5 x 5 x 48, kernels = 256, stride = 1
# 3rd layer -> dim = 5 x 5 x 48, ouptut dim -> 3 x 3 x 256, kernels = 348, stride = 1
# 4th layer -> dim = 3 x 3 x 256, output dim -> 3 x 3 x 192, kernels = 348, stride = 1
# 5th layer -> dim = 3 x 3 x 192, output dim -> 3 x 3 x 192, kernels = 256, stride = 1

import torch
import torch.nn as nn
from torchsummary import summary

class AlexNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, padding, pool_and_norm):
        super(AlexNetBlock, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride = stride, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.pool_and_norm = pool_and_norm
        if pool_and_norm:
            self.norm_layer = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75, k = 2)
            self.pool_layer = nn.MaxPool2d(stride = 2, kernel_size=3)   # During pooling, different kernel_size and stride led to lesser error rates
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.relu(x)
        if self.pool_and_norm:
            x = self.norm_layer(x)
            x = self.pool_layer(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes, in_channels) -> None:
        super(AlexNet, self).__init__()
        #Convolutional part (first 5 layers)
        self.block1 = AlexNetBlock(in_channels=3, out_channels=96, stride=4, kernel_size=11, padding=0, pool_and_norm=True)
        self.block2 = AlexNetBlock(in_channels=96, out_channels=256, stride=1, kernel_size=5, padding = 2, pool_and_norm=True)
        self.block3 = AlexNetBlock(in_channels = 256, out_channels=348, stride = 1, kernel_size=3, padding = 1, pool_and_norm=False)
        self.block4 = AlexNetBlock(in_channels = 348, out_channels = 348, stride = 1, kernel_size=3, padding = 1, pool_and_norm=False)
        self.block5 = AlexNetBlock(in_channels = 348, out_channels=256, stride = 1, kernel_size=3, padding=1, pool_and_norm=True)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.dropout1 = nn.Dropout(0.5) # Probability for neuron to not participate is 0.5 according to the paper
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.classification_layer = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)


        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.classification_layer(x)
        return x
        
if __name__ == "__main__":
    net = AlexNet(num_classes = 10, in_channels = 3)
    summary(net, (3, 227, 227))