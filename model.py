import torch
import torch.nn as nn
import torch.nn.functional as F


def get_pad(kernel_size):

    if isinstance(kernel_size, int):
        return kernel_size // 2
    elif isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
        return (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        raise ValueError(f"Not supported type for kernel_size: {type(kernel_size)}. "
                         f"Supported types: int, list, tuple")


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), strides=(1, 1)):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels[0], out_channels[0], kernel_size[0],
                               strides[0], padding="same")
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(in_channels[1], out_channels[1], kernel_size[1],
                               strides[1], padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels[1])

        if in_channels[0] != out_channels[1]:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels[0], out_channels[1], 1, strides[0], bias=False),
                nn.BatchNorm2d(out_channels[1])
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.prelu(self.bn1(self.conv1(x)), torch.tensor(0.25, device='cuda'))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.prelu(out, torch.tensor(0.25, device='cuda'))


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding="same", activation="lrelu", bn=True):
        super(ConvLayer, self).__init__()
        if padding == "same" and stride > 1:
            padding = get_pad(kernel_size)
        self.activation = activation
        self.bn = bn
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.batch_norm(x)
        if self.activation == "lrelu":
            x = F.leaky_relu(x, 0.2)
        elif self.activation == "prelu":
            x = F.prelu(x, torch.tensor(0.25, device='cuda'))
        return x


class ConvLayerPixelShuffler(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayerPixelShuffler, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=get_pad(kernel_size))
        self.pixelShuffler = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixelShuffler(x)
        x = F.prelu(x, torch.tensor(0.25, device='cuda'))
        return x


class Discriminator(nn.Module):
    def __init__(self, height, width):
        super(Discriminator, self).__init__()
        self.conv1 = ConvLayer(3, 64, 3, 1, bn=False)
        self.conv2 = ConvLayer(64, 64, 3, 2)
        self.conv3 = ConvLayer(64, 128, 3, 1)
        self.conv4 = ConvLayer(128, 128, 3, 2)
        self.conv5 = ConvLayer(128, 256, 3, 1)
        self.conv6 = ConvLayer(256, 256, 3, 2)
        self.conv7 = ConvLayer(256, 512, 3, 1)
        self.conv8 = ConvLayer(512, 512, 3, 2)
        self.fc1 = nn.Linear(int(512*(height/2**4)*(width/2**4)), 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, image):
        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        logits = self.fc2(x)
        output = torch.sigmoid(logits)
        return output


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = ConvLayer(3, 64, 9, 1, activation="prelu", bn=False)
        self.res1 = ResBlock([64, 64], [64, 64])
        self.res2 = ResBlock([64, 64], [64, 64])
        self.res3 = ResBlock([64, 64], [64, 64])
        self.res4 = ResBlock([64, 64], [64, 64])
        self.res5 = ResBlock([64, 64], [64, 64])
        self.conv2 = ConvLayer(64, 64, 3, 1, activation=None)
        self.conv3 = ConvLayerPixelShuffler(64, 256, 3, 1)
        # Number of output channels after pixel shuffler is reduced 4 times
        self.conv4 = ConvLayerPixelShuffler(64, 256, 3, 1)
        self.conv5 = nn.Conv2d(64, 3, 9, 1, padding="same")

    def forward(self, image):
        conv1 = self.conv1(image)
        x = self.res1(conv1)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.conv2(x)
        x = conv1 + x
        x = self.conv3(x)
        x = self.conv4(x)
        output = self.conv5(x)
        return output
