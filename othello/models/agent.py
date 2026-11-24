"""SHA-ZA policy/value networks (OthelloRL) and the separate value network."""
import torch
from torch import nn
from torch.nn import Module

from othello.models.blocks import ResidualBlock


class value_Net(nn.Module):
    """Standalone value head: 3-channel input (board, under-attack, policy output) -> score in [-1, 1]."""

    def __init__(self):
        super(value_Net, self).__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3),
                      padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=64, affine=True),
            nn.LeakyReLU()
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(channels=64),
            ResidualBlock(channels=64),
            ResidualBlock(channels=64),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.fc(self.res_blocks(self.conv_in(x)))


class OthelloRL(Module):
    """Othello agent: policy head (masked softmax over squares) + optional value head."""

    def __init__(self, internal_channels=64, res_depth=8, name="A",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), return_val=False):
        super(OthelloRL, self).__init__()

        # Input convolutional layer
        self.InConv = nn.Sequential(
            # input is board and opponent's pieces under attack
            nn.Conv2d(in_channels=2, out_channels=internal_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=internal_channels, affine=True),
            nn.LeakyReLU(inplace=True)
        )

        # ResidualBlock block
        self.ResidualBlock = nn.Sequential(
            *[ResidualBlock(channels=internal_channels) for _ in range(res_depth)]
        )

        # Output convolutional layer
        self.OutConv = nn.Sequential(
            nn.Conv2d(in_channels=internal_channels, out_channels=1,
                      kernel_size=(1, 1), bias=True)
        )

        self.value_net = value_Net()  # value network

        self.internal_channels = internal_channels
        self.res_depth = res_depth
        self.name = name
        self.return_val = return_val
        self.device = device

        self.softmax = nn.Softmax(dim=1)

    ##############################
    def forward(self, input, mask):
        x = input
        x = self.InConv(x)
        x = self.ResidualBlock(x)
        x = self.OutConv(x)
        # Apply masking to prevent choosing already chosen positions
        x = x * mask + (1 - mask) * -1e32  # Set chosen positions to a large negative value
        x = self.softmax(x.view(x.size(0), -1)).view(x.size())

        if self.return_val:
            value = self.value_net(torch.cat((input, x), axis=1))
            return x, value

        return x

    ###############################
    def save(self, path):
        # function returns true when object is saved, false when object isn't saved
        try:
            torch.save(self, path)
            return True
        except Exception as e:
            print("Model Saving Failed :", e)
            return False

    def load(self, path):
        # load model from path otherwise return false and print error message
        try:
            loaded_model = torch.load(path, map_location=self.device)
            self.__dict__.update(loaded_model.__dict__)  # Update current instance with loaded model's attributes
            return True

        except Exception as e:
            print(f"Model {self.name} wasn't found :", e)
            return False
