import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf, DictConfig

class ResNet(nn.Module):
    def __init__(self, model_cfg: DictConfig, device):
        super().__init__()

        OBSERVATION_WIDTH = 5
        ACTION_SIZE = 4

        SCALAR_FEATURES_SIZE = 3*model_cfg.history_length + 2  # see Maze.get_encoded_scalar_features
        num_filters = model_cfg.num_filters
        num_resBlocks = model_cfg.num_resBlocks
        self.device = device


        # Initial convolutional block
        # Input channels are stack of (all observation planes (hist len), all target planes (hist len), all action planes (hist len - 1))
        self.startBlock = nn.Sequential(
            nn.Conv2d(in_channels=3 * model_cfg.history_length - 1, out_channels=num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU()
        )

        # Residual blocks
        self.backBone = nn.ModuleList(
            [ResBlock(num_filters) for _ in range(num_resBlocks)]
        )

        # Policy head convolutional part that gets flattened
        self.policyHead_conv = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the size after flattening
        policy_conv_output_size = 32 * OBSERVATION_WIDTH ** 2

        # Policy head fully connected part
        self.policyHead_flat = nn.Sequential(
            nn.Linear(policy_conv_output_size + SCALAR_FEATURES_SIZE, 256),  # Adding scalar features
            nn.ReLU(),
            nn.Linear(256, ACTION_SIZE),
        )

        # Value head convolutional part
        self.valueHead_conv = nn.Sequential(
            nn.Conv2d(num_filters, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the size after flattening
        value_conv_output_size = 3 * OBSERVATION_WIDTH ** 2

        # Value head fully connected part
        self.valueHead_flat = nn.Sequential(
            nn.Linear(value_conv_output_size + SCALAR_FEATURES_SIZE, 256), # Adding scalar features
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh() # Value is between -1 and 1
        )

        self.to(device)

    def forward(self, x, scalar_features):
        # x: Input tensor of shape (batch_size, 2 * history_length, maze_height, maze_width)
        # scalar_features: (batch_size, 3 * history length + 2), normalized

        # Initial convolutional block
        x = self.startBlock(x)

        # Residual blocks
        for resBlock in self.backBone:
            x = resBlock(x)

        # Policy head
        policy_x = self.policyHead_conv(x)  # Output is already flattened
        # Concatenate positions
        policy_x_concat = torch.cat([policy_x, scalar_features], dim=1)
        policy = self.policyHead_flat(policy_x_concat)

        # Value head
        value_x = self.valueHead_conv(x)  # Output is already flattened
        # Concatenate positions
        value_x_concat = torch.cat([value_x, scalar_features], dim=1)
        value = self.valueHead_flat(value_x_concat)

        return policy, value

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
