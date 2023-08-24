import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = 3
        out_channels = 3
        kernel_size = 3

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding='same'),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding='same'),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU())
        self.af = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        out = self.conv2(x)
        out = out + residual
        out = self.af(out)
        return out


class CNN(nn.Module):
    def __init__(self, classes):
        super().__init__()

        out_channels = 3

        self.Start = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same')
        self.Res1 = ResidualBlock()
        self.Res2 = ResidualBlock()
        self.Res3 = ResidualBlock()
        self.Res4 = ResidualBlock()
        self.spp1 = nn.AdaptiveMaxPool2d((1, 1))
        self.spp2 = nn.AdaptiveMaxPool2d((2, 2))
        self.spp3 = nn.AdaptiveMaxPool2d((4, 4))

        self.fc = nn.Linear(21 * out_channels, classes)

    def forward(self, hidden_states):
        x = hidden_states.unsqueeze(1)
        x = self.Start(x)
        x = self.Res1(x)
        x = self.Res2(x)
        x = self.Res3(x)
        x = self.Res4(x)
        spp1 = self.spp1(x)
        spp1 = spp1.flatten(1)
        spp2 = self.spp2(x)
        spp2 = spp2.flatten(1)
        spp3 = self.spp3(x)
        spp3 = spp3.flatten(1)
        spp = torch.cat((spp1, spp2, spp3), 1)
        out = self.fc(spp)

        return out
