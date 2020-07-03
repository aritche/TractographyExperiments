"""
A model for generating streamlines for a single tract
This is an adaptation of HairNet (Zhou et al. 2018) https://doi.org/10.1007/978-3-030-01252-6_15
"""
import os
import numpy as np
import cv2

import torch
import torch.nn as nn
from torchsummary import summary

class CustomModel(nn.Module):
    def __init__(self):
        super(SingleTractModel, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Encoding (input -> 512 vector)
        #self.conv1 = torch.nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(9,9,9), stride=2, padding=3)
        self.down_conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=2, padding=3)
        self.down_conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=8, stride=2, padding=3)
        self.down_conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=6, stride=2, padding=2)
        self.down_conv_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.down_conv_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.down_conv_6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.down_conv_7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.down_pool_8 = nn.MaxPool2d(kernel_size=8)

        # Decoding (512 vector -> 32 x 32 x 512 volume)
        self.up_linear_1 = nn.Linear(in_features=512, out_features=1024)
        self.up_linear_2 = nn.Linear(in_features=1024, out_features=4096)
        self.up_conv_3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.up_conv_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.up_conv_5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        # Position decoder (32 x 32 x 512 volume -> 32 x 32 x 300)
        self.pos_conv_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.pos_conv_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.pos_conv_3 = nn.Conv2d(in_channels=512, out_channels=100, kernel_size=1, stride=1, padding=0)

        # Curvature decoder (32 x 32 x 512 volume -> 32 x 32 x 100)
        self.curv_conv_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.curv_conv_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.curv_conv_3 = nn.Conv2d(in_channels=512, out_channels=300, kernel_size=1, stride=1, padding=0)

        # Seed decoder (32 x 32 x 512 volume -> 32 x 32 x 3)
        #self.seed_conv_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        #self.seed_conv_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        #self.seed_conv_3 = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.down_conv_1(x)
        x = self.relu(x)
        x = self.down_conv_2(x)
        x = self.relu(x)
        x = self.down_conv_3(x)
        x = self.relu(x)
        x = self.down_conv_4(x)
        x = self.relu(x)
        x = self.down_conv_5(x)
        x = self.relu(x)
        x = self.down_conv_6(x)
        x = self.relu(x)
        x = self.down_conv_7(x)
        x = self.relu(x)
        x = self.down_pool_8(x)
        x = self.tanh(x)

        x = self.up_linear_1(x)
        x = self.relu(x)
        x = self.up_linear_2(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.up_conv_3(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.up_conv_4(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.up_conv_5(x)
        x = self.relu(x)

        c = self.curv_conv_1(x)
        c = self.relu(c)
        c = self.curv_conv_2(c)
        c = self.tanh(c)
        c = self.curv_conv_3(c)

        p = self.pos_conv_1(x)
        p = self.relu(p)
        p = self.pos_conv_2(p)
        p = self.tanh(p)
        p = self.pos_conv_3(p)

        return [c, p]

# Custom loss function
def CustomLoss(output, target):
    curvature_output, position_output = output
    curvature_target, position_target = target

    criterion = nn.MSELoss()

    position_loss  = criterion(curvature_output, curvature_target)
    curvature_loss = criterion(position_output, position_target)

    return position_loss + curvature_loss

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir):
        self.data_files = {}

        data_files[subject]
        self.data_files = os.listdir()
        self.data_files.sort()

    # Given an index, return the loaded [data, label]
    def __getitem__(self, idx):

    def __len__(self):
