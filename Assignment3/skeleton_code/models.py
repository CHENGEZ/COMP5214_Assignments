#  COMP 6211D & ELEC 6910T , Assignment 3
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator       --> Used in the vanilla GAN in Part 1
#   - CycleGenerator    --> Used in the CycleGAN in Part 2
#   - DCDiscriminator   --> Used in both the vanilla GAN and CycleGAN (Parts 1 and 2)
#
# For the assignment, you are asked to create the architectures of these three networks by
# filling in the __init__ methods in the DCGenerator, CycleGenerator, and DCDiscriminator classes.
# Note that the forward passes of these models are provided for you, so the only part you need to
# fill in is __init__.

import torch
import torch.nn as nn
import torch.nn.functional as F


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class DCGenerator(nn.Module):
    def __init__(self, noise_size, conv_dim):
        super(DCGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################
        self.deconv1 = deconv(in_channels=100, out_channels=128,stride=1, kernel_size=4, padding=0, batch_norm=True)
        self.deconv2 = deconv(in_channels=128, out_channels=64, stride=1, kernel_size=5, padding=0, batch_norm=True)
        self.deconv3 = deconv(in_channels=64, out_channels=32, stride=2, kernel_size=4, padding=1, batch_norm=True)
        self.deconv4 = deconv(in_channels=32, out_channels=3, stride=2, kernel_size=4, padding=1, batch_norm=False)
            
    def forward(self, z):
        """Generates an image given a sample of random noise.

            Input
            -----
                z: BS x noise_size x 1 x 1   -->  16x100x1x1

            Output
            ------
                out: BS x channels x image_width x image_height  -->  16x3x32x32
        """

        out = F.relu(self.deconv1(z))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = F.tanh(self.deconv4(out))
        return out


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """

    def __init__(self):
        super(CycleGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # 1. Define the encoder part of the generator (that extracts features from the input image)
        self.conv1 = conv(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2, batch_norm=True)
        self.conv2 = conv(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, batch_norm=True)
        # 2. Define the transformation part of the generator
        self.resnet_block1 = ResnetBlock(128)
        self.resnet_block2 = ResnetBlock(128)
        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.deconv1 = deconv(in_channels=128, out_channels=64, stride=2, kernel_size=4, padding=1, batch_norm=True)
        self.deconv2 = deconv(in_channels=64, out_channels=3, stride=2, kernel_size=4, padding=1, batch_norm=False)

    def forward(self, x):
        """Generates an image conditioned on an input image.

            Input
            -----
                x: BS x 3 x 32 x 32

            Output
            ------
                out: BS x 3 x 32 x 32
        """

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out = F.relu(self.resnet_block1(out))
        out = F.relu(self.resnet_block2(out))

        out = F.relu(self.deconv1(out))
        out = F.tanh(self.deconv2(out))

        return out


class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """

    def __init__(self, conv_dim=32):
        super(DCDiscriminator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################
        self.conv1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=5, stride=2, padding=2, batch_norm=True)
        self.conv2 = conv(in_channels=conv_dim, out_channels=2*conv_dim, kernel_size=5, stride=2, padding=2, batch_norm=True)
        self.conv3 = conv(in_channels=2*conv_dim, out_channels=4*conv_dim, kernel_size=5, stride=2, padding=2, batch_norm=True)
        self.conv4 = conv(in_channels=4*conv_dim, out_channels=1, kernel_size=5, stride=2, padding=1, batch_norm=False)

    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = self.conv4(out).squeeze()
        out = F.sigmoid(out)
        return out
