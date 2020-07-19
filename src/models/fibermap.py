"""
A model for generating streamlines for a single tract
Uses 3D TOM input
This is an adaptation of HairNet (Zhou et al. 2018) https://doi.org/10.1007/978-3-030-01252-6_15
"""
import os
import numpy as np
import random
import cv2
import nibabel as nib
from glob import glob

import torch
import torch.nn as nn

from dipy.io.streamline import load_trk
from dipy.tracking.streamline import set_number_of_points, select_random_set_of_streamlines
from nibabel import trackvis

points_per_sl = 15
num_sl = 5

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
                                                                                                                 # VOLUME SIZE                      # PARAMETERS
        # Encoding (input -> 512 vector)                                                                         # 3 x 144 x 144 x 144 -> 8.9M      (IN * F^3 + 1)*OUT
        self.down_conv_1 = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(9,9,9), stride=1, padding=0)     # K   x 136 x 136 x 136 ->        3*(3*9^3+1)     = 6.5K
        self.down_conv_2 = nn.Conv3d(in_channels=3, out_channels=44, kernel_size=(10,10,10), stride=2, padding=0) # K   x 64  x 64  x 64  ->        44*(3*10^3+1)   = 132K
        self.down_conv_3 = nn.Conv3d(in_channels=44, out_channels=32, kernel_size=(6,6,6), stride=2, padding=2)   # K   x 32  x 32  x 32  ->        32*(44*6^3+1)   = 305K
        self.down_conv_4 = nn.Conv3d(in_channels=32, out_channels=76, kernel_size=(6,6,6), stride=2, padding=2)   # K   x 16  x 16  x 16  ->        76*(32*6^3+1)   = 525K
        self.down_conv_5 = nn.Conv3d(in_channels=76, out_channels=288, kernel_size=(3,3,3), stride=1, padding=1)  # K   x 16  x 16  x 16  ->        288*(76*3^3+1)  = 590K
        self.down_conv_6 = nn.Conv3d(in_channels=288, out_channels=150, kernel_size=(4,4,4), stride=2, padding=1) # K   x 8   x 8   x 8   ->        150*(288*4^3+1) = 2.7M
        self.down_conv_7 = nn.Conv3d(in_channels=150, out_channels=512, kernel_size=(3,3,3), stride=1, padding=1) # K   x 8   x 8   x 8   ->        512*(150*3^3+1) = 2.1M
        self.down_pool_8 = nn.MaxPool3d(kernel_size=8)                                                            # 512 x 1   x 1   x 1                  

        # Decoding (512 vector -> 32 x 32 x 256 volume)
        self.up_linear_1 = nn.Linear(in_features=512, out_features=1024)
        self.up_linear_2 = nn.Linear(in_features=1024, out_features=4096)
        self.up_conv_3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.up_conv_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.up_conv_5 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)

        # Reducing down to 30 x 30 x 15
        self.fiber_map = nn.Conv2d(in_channels=256, out_channels=15, kernel_size=3, stride=1, padding=0)

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

        print(x.size())
        x = x.view(-1, 512)

        x = self.up_linear_1(x)
        x = self.relu(x)
        x = self.up_linear_2(x)
        x = self.relu(x)

        x = x.view(-1, 1, 64, 64)
        x = self.upsample(x)

        x = x.view(-1, 256, 8, 8) # 8 x 8
        x = self.up_conv_3(x)     # 8 x 8
        x = self.relu(x)
        x = self.upsample(x)      # 16 x 16
        x = self.up_conv_4(x)     # 16 x 16
        x = self.relu(x)
        x = self.upsample(x)      # 32 x 32
        x = self.up_conv_5(x)
        x = self.relu(x)

        x = self.fiber_map(x)

        return x

def generate_fibermaps(streamlines):
    forward_points = streamlines
    backward_points = [item[::-1] for item in forward_points]

    fiber_maps = []
    for i in range(len(streamlines)):
        rowA = np.append(forward_points[i], backward_points[i], axis=0)
        rowB = np.append(backward_points[i], forward_points[i], axis=0)
        fiber_map = np.zeros((points_per_sl*2,points_per_sl*2,3))
        for row in range(points_per_sl*2):
            if (row % 2 == 0):
                fiber_map[row] = rowA
            else:
                fiber_map[row] = rowB
        fiber_maps.append(fiber_map)

    return fiber_maps

# Custom loss function
def CustomLoss(output, target):
    # Re-implemented MSE loss for efficiency reasons
    return ((output - target)**2).mean()

def get_data(in_fn, out_fn, mean, sdev):
    # Load TOM and preprocess 
    tom = nib.load(in_fn).get_data() # 144 x 144 x 144 x 3
    tom = (tom - mean) / sdev # normalise based on dataset mean/stdev
    tom = torch.from_numpy(np.float32(tom))
    tom = tom.permute(3, 0, 1, 2) # channels first for pytorch
    
    # Load the tractogram
    streamlines, header = trackvis.read(out_fn)
    streamlines = [s[0] for s in streamlines]

    # Sort streamlines by mid points x, then y, then z
    mids = [sl[len(sl)//2].copy() for sl in streamlines]
    streamlines = [x for _, x in sorted(zip(mids, streamlines), key=lambda pair: [pair[0][0], pair[0][1], pair[0][2]])]

    # Convert tractogram to FiberMaps
    fiber_maps = generate_fibermaps(streamlines)

    # Visualise the FiberMaps to ensure they have been generated correctly
    """
    cv2.namedWindow('map', cv2.WINDOW_NORMAL)
    for fm in fiber_maps:
        abs_map = fm
        abs_map = (abs_map - np.min(abs_map)) / (np.max(abs_map) - np.min(abs_map))
        abs_map *= 255
        cv2.imshow('map', np.uint8(abs_map))
        cv2.waitKey(0)
    """

    # Convert fibermap to torch
    fiber_maps = np.array(fiber_maps)
    fiber_maps = np.transpose(fiber_maps, (1,2,3,0))
    # Merge the final 2 dimensions
    #fiber_maps = np.reshape(fiber_maps, fiber_maps.shape[:-2] + (-1,))
    #print(fiber_maps.shape)
    fiber_maps = np.reshape(fiber_maps, (points_per_sl*2, points_per_sl*2, num_sl*3))
    fiber_maps = torch.from_numpy(fiber_maps)
    fiber_maps = fiber_maps.permute(2, 0, 1) # channels first for pytorch

    return [tom, fiber_maps]

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, toms_dir, beginnings_dir, endings_dir, tractograms_dir, means=np.array([]), sdevs=np.array([])):
            
        # Store all input filenames
        self.input_files = glob(toms_dir + '/*.nii.gz')

        # Calculate mean and standard deviation of the input dataset
        if len(means) == 0:
            self.means = np.float32(np.array([0, 0, 0]))
            for fn in self.input_files:
                data = nib.load(fn).get_data()
                self.means += np.mean(data.reshape((-1,3)), axis=0)
            self.means = self.means/(len(self.input_files))


            self.sdevs = np.float32(np.array([0, 0, 0]))
            squared_diffs_sum = np.float32(np.array([0,0,0]))
            for fn in self.input_files:
                data = nib.load(fn).get_data()
                pixels = data.reshape((-1,3))
                squared_diffs_sum += np.sum((pixels - self.means)**2, axis=0)
            self.sdevs  = (squared_diffs_sum / (len(self.input_files) * 144*144*144))**(1/2)

        else:
            self.means = means
            self.sdevs  = sdevs
        print("NORMALISING WITH MEANS: %f, %f, %f" % (self.means[0], self.means[1], self.means[2]))
        print("NORMALISING WITH SDEVS: %f, %f, %f" % (self.sdevs[0], self.sdevs[1], self.sdevs[2]))

        # Store all output files
        self.output_files = glob(tractograms_dir + '/*.trk')

        # Sort for correct matching between the two sets of filenames
        self.input_files.sort()
        self.output_files.sort()

    # Given an index, return the loaded [data, label]
    def __getitem__(self, idx):
        #return get_data(self.input_files[idx], self.endings_files[idx], self.output_files[idx])
        return get_data(self.input_files[idx], self.output_files[idx], self.means, self.sdevs)

    def __len__(self):
        return len(self.input_files)

def OutputToStreamlines(output):
    # This could probably be sped up if using torch operations e.g. https://stackoverflow.com/questions/55757255/replicate-subtensors-in-pytorch
    fiber_maps = output.permute(1, 2, 0) # (5*3, 30, 30) -> (30, 30, 5*3)
    fiber_maps = fiber_maps.cpu().detach().numpy()
    fiber_maps = np.reshape(fiber_maps, (points_per_sl*2, points_per_sl*2, num_sl, 3))
    fiber_maps = np.moveaxis(fiber_maps, [0, 2, 2], [2, 1, 0])

   # for fiber_map in fiber_maps:
        
    
    return streamlines
