import torch
import torch.nn as nn
import torch.nn.functional as F

class head_branch(nn.Module):
    def __init__(self, in_channels):
        super(head_branch, self).__init__()
        self.theta = nn.Sequential(
            nn.Conv1d(in_channels, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.Conv1d(512, in_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.theta(x)


class BottleNeck(nn.Module):
    def __init__(self, in_channel):
        super(BottleNeck, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv1d(in_channel, 512, kernel_size=1),
            # nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv1d(512, in_channel, kernel_size=1),
            # nn.ReLU()
        )

    def forward(self, x):
        return x + self.conv(x)

class FE_Net(nn.Module):
    def __init__(self, in_channel, btn=3):
        super(FE_Net, self).__init__()
        self.conv = nn.Sequential(*[BottleNeck(in_channel) for _ in range(btn)])
        self.theta = head_branch(in_channel)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        mu = self.conv(x)
        theta = self.theta(x)
        mu = mu.permute(0, 2, 1)
        theta = theta.permute(0, 2, 1)
        return mu, -theta