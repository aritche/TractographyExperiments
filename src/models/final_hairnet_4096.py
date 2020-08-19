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
from dipy.tracking import utils

num_points = 40
num_streamlines = 1024
w_coords = 1 # weight for the coords loss
w_seeds =  1 # weight for the seeds loss

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = torch.nn.Dropout(p=0.50)
                                                                                                                 # VOLUME SIZE                      # PARAMETERS
        # Encoding (input -> 512 vector)                                                                         # 3 x 144 x 144 x 144 -> 8.9M      (IN * F^3 + 1)*OUT

        self.down_conv_1 = nn.Conv3d(in_channels=5, out_channels=32, kernel_size=(9,9,9), stride=1, padding=0)     # K   x 136 x 136 x 136 ->        3*(3*9^3+1)     = 6.5K
        self.batchnorm_1 = nn.BatchNorm3d(32)

        self.down_conv_2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(10,10,10), stride=2, padding=0) # K   x 64  x 64  x 64  ->        44*(3*10^3+1)   = 132K
        self.batchnorm_2 = nn.BatchNorm3d(64)

        self.down_conv_3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(6,6,6), stride=2, padding=2)   # K   x 32  x 32  x 32  ->        32*(44*6^3+1)   = 305K
        self.batchnorm_3 = nn.BatchNorm3d(128)

        self.down_conv_4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(6,6,6), stride=2, padding=2)   # K   x 16  x 16  x 16  ->        76*(32*6^3+1)   = 525K
        self.batchnorm_4 = nn.BatchNorm3d(256)

        self.down_conv_5 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3,3,3), stride=1, padding=1)  # K   x 16  x 16  x 16  ->        288*(76*3^3+1)  = 590K
        self.batchnorm_5 = nn.BatchNorm3d(256)

        self.down_conv_6 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(4,4,4), stride=2, padding=1) # K   x 8   x 8   x 8   ->        150*(288*4^3+1) = 2.7M
        self.batchnorm_6 = nn.BatchNorm3d(256)

        self.down_conv_7 = nn.Conv3d(in_channels=256, out_channels=4096, kernel_size=(3,3,3), stride=1, padding=1) # K   x 8   x 8   x 8   ->        512*(150*3^3+1) = 2.1M
        self.batchnorm_7 = nn.BatchNorm3d(4096)

        self.down_pool_8 = nn.MaxPool3d(kernel_size=8)                                                            # 512 x 1   x 1   x 1                  
        self.batchnorm_8 = nn.BatchNorm3d(4096)

        # Decoding (512 vector -> 32 x 32 x 512 volume)
        #self.up_linear_1 = nn.Linear(in_features=512, out_features=1024)
        #self.up_linear_1 = nn.Linear(in_features=2048, out_features=4096)
        self.up_conv_3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.batchnorm_9 = nn.BatchNorm2d(512)
        self.up_conv_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.batchnorm_10 = nn.BatchNorm2d(512)
        self.up_conv_5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.batchnorm_11 = nn.BatchNorm2d(512)

        # Position decoder (32 x 32 x 512 volume -> 32 x 32 x 300)
        self.pos_conv_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.batchnorm_12 = nn.BatchNorm2d(512)
        self.pos_conv_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.batchnorm_13 = nn.BatchNorm2d(512)
        self.pos_conv_3 = nn.Conv2d(in_channels=512, out_channels=num_points*3, kernel_size=1, stride=1, padding=0)

        # Seed decoder (32 x 32 x 512 volume -> 32 x 32 x 3)
        #self.seed_conv_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        #self.batchnorm_14 = nn.BatchNorm2d(512)
        #self.seed_conv_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        #self.batchnorm_15 = nn.BatchNorm2d(512)
        #self.seed_conv_3 = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x, seeds):
        # Input: 3 x 144 x 144 x 144
        x = self.down_conv_1(x) # Output: (3, 136, 136, 136)
        x = self.batchnorm_1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.down_conv_2(x) # Output: (44, 64, 64, 64)
        x = self.batchnorm_2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.down_conv_3(x) # Output: (32, 32, 32, 32)
        x = self.batchnorm_3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.down_conv_4(x) # Output: (76, 16, 16, 16)
        x = self.batchnorm_4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.down_conv_5(x) # Output: (288, 16, 16, 16 )
        x = self.batchnorm_5(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.down_conv_6(x) # Output: (150, 8, 8, 8)
        x = self.batchnorm_6(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.down_conv_7(x) # Output: (512, 8, 8, 8)
        x = self.batchnorm_7(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.down_pool_8(x) # Output: (512, 1, 1, 1)
        x = self.batchnorm_8(x)
        x = self.tanh(x)
        x = self.dropout(x)

        x = x.view(-1, 4096)     # Output: (2048)

        #x = self.up_linear_1(x) # Output: (1024)
        #x = self.relu(x)
        #x = self.dropout(x)

        #x = self.up_linear_2(x) # Output: (4096)
        #x = self.relu(x)
        #x = self.dropout(x)

        x = x.view(-1, 256, 4, 4) # Output: (256, 4, 4)
        x = self.upsample(x)      # Output: (256, 8, 8)

        x = self.up_conv_3(x)     # Output: (512, 8, 8)
        x = self.batchnorm_9(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.upsample(x)      # Output: (512, 16, 16)

        x = self.up_conv_4(x)     # Output: (512, 16, 16)
        x = self.batchnorm_10(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.upsample(x)      # Output: (512, 32, 32)

        x = self.up_conv_5(x)     # Output: (512, 32, 32)
        x = self.batchnorm_11(x)
        x = self.relu(x)
        x = self.dropout(x)

        p = self.pos_conv_1(x)    # Output: (512, 32, 32)
        p = self.batchnorm_12(p)
        p = self.relu(p)
        p = self.dropout(p)

        p = self.pos_conv_2(p)    # Output: (512, 32, 32)
        p = self.batchnorm_13(p)
        p = self.tanh(p)
        p = self.dropout(p)

        p = self.pos_conv_3(p)    # Output: (num_points*3, 32, 32)

        #s = self.seed_conv_1(x)    # Output: (512, 32, 32)
        #s = self.batchnorm_14(s)
        #s = self.relu(s)          
        #s = self.dropout(s)

        #s = self.seed_conv_2(s)    # Output: (512, 32, 32)
        #s = self.batchnorm_15(s)
        #s = self.tanh(s)
        #s = self.dropout(s)

        #s = self.seed_conv_3(s)    # Output: (3, 32, 32)

        return p

# Custom loss function
def CustomLoss(output, target):
    # Re-implemented MSE loss for efficiency reasons
    return ((output - target)**2).mean()

def get_data(tom_fn, tractogram_fn, beginnings_fn, endings_fn, mean, sdev, coord2voxel):
    #print('-----')

    # Load TOM volume and preprocess
    tom = nib.load(tom_fn).get_data() # 144 x 144 x 144 x 3
    tom = (tom - mean) / sdev # normalise based on dataset mean/stdev
    tom = torch.from_numpy(np.float32(tom))
    tom = tom.permute(3, 0, 1, 2) # channels first for pytorch
    
    # On-the-fly augmentation
    noise_stdev = torch.rand(1) * 0.05
    noise = torch.normal(mean=torch.zeros(tom.size()), std=torch.ones(tom.size())*noise_stdev)
    tom += noise

    # Load the beginnings segmentation volume
    beginnings = nib.load(beginnings_fn).get_data() # 144 x 144 x 144 x 3
    beginnings = torch.from_numpy(np.float32(beginnings))
    beginnings = beginnings.permute(3, 0, 1, 2) # channels first for pytorch

    # Load the endings segmentation volume
    endings = nib.load(endings_fn).get_data() # 144 x 144 x 144 x 3
    endings = torch.from_numpy(np.float32(endings))
    endings = endings.permute(3, 0, 1, 2) # channels first for pytorch

    # Concatenate the seed volume as an extra channel of the first dimension of the TOM volume
    tom_seed = torch.cat((tom, beginnings), dim=0)
    tom_seed = torch.cat((tom_seed, endings), dim=0)

    # Load the tractogram
    #print('Loading tractogram')
    #t0 = time.time()
    streamlines, header = trackvis.read(tractogram_fn)
    streamlines = [s[0] for s in streamlines]
    streamlines = np.array(streamlines)

    # Convert to voxel coordinates and normalise
    streamlines = utils.apply_affine(aff=coord2voxel, pts=streamlines)
    streamlines /= 144

    # Get seed coordinates and remove from streamlines
    seeds = np.array([sl[0].copy() for sl in streamlines], dtype=np.float32)
    for i in range(len(streamlines)):
        streamlines[i] -= streamlines[i][0]

    # Sort seeds and streamlines by seed points x, then y, then z
    streamlines = list(streamlines)
    streamlines = [x for _, x in sorted(zip(seeds, streamlines), key=lambda pair: [pair[0][0], pair[0][1], pair[0][2]])]
    seeds = sorted(seeds, key=lambda k: [k[0], k[1], k[2]])
    seeds = np.array(seeds, dtype=np.float32)

    # Convert seeds to torch
    seeds = torch.from_numpy(seeds)
    seeds = seeds.permute(1,0) # channels first for pytorch

    # automatically converts list to numpy array and reshapes it
    # (num_sl, points_per_sl, 3) -> (sqrt(num_sl), sqrt(num_sl), points_per_sl*3)
    # Performed in 2 successive steps because I don't know if it works if I do it in one step
    streamlines = np.reshape(streamlines, (int(num_streamlines**(1/2)), int(num_streamlines**(1/2)), num_points, 3))
    streamlines = np.reshape(streamlines, (int(num_streamlines**(1/2)), int(num_streamlines**(1/2)), num_points*3))
    tractogram = torch.from_numpy(streamlines)
    tractogram = tractogram.permute(2, 0, 1) # channels first for pytorch

    return [[tom_seed, seeds], tractogram]

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, toms_dir, tractograms_dir, beginnings_dir, endings_dir, coord2voxel, means=np.array([]), sdevs=np.array([])):
        # Store affine transformation
        self.coord2voxel = coord2voxel

        # Get lists of files
        self.toms = glob(toms_dir + '/*.nii.gz')
        self.beginnings = glob(beginnings_dir + '/*.nii.gz')
        self.endings = glob(endings_dir + '/*.nii.gz')
        self.tractograms = glob(tractograms_dir + '/*.trk')

        # Calculate mean and standard deviation of the input dataset
        if len(means) == 0:
            print("Calculating mean for the dataset...")
            self.means = np.float32(np.array([0, 0, 0]))
            for fn in self.toms:
                data = nib.load(fn).get_data()
                self.means += np.mean(data.reshape((-1,3)), axis=0)
            self.means = self.means/(len(self.toms))

            print("Calculating sdev for the dataset...")
            self.sdevs = np.float32(np.array([0, 0, 0]))
            squared_diffs_sum = np.float32(np.array([0,0,0]))
            for fn in self.toms:
                data = nib.load(fn).get_data()
                pixels = data.reshape((-1,3))
                squared_diffs_sum += np.sum((pixels - self.means)**2, axis=0)
            self.sdevs  = (squared_diffs_sum / (len(self.toms) * 144*144*144))**(1/2)

        else:
            self.means = means
            self.sdevs  = sdevs
        print("NORMALISING WITH MEANS: %f, %f, %f" % (self.means[0], self.means[1], self.means[2]))
        print("NORMALISING WITH SDEVS: %f, %f, %f" % (self.sdevs[0], self.sdevs[1], self.sdevs[2]))

        # Sort for correct matching between the sets of filenames
        self.toms.sort()
        self.beginnings.sort()
        self.endings.sort()
        self.tractograms.sort()

    # Given an index, return the loaded [data, label]
    def __getitem__(self, idx):
        return get_data(self.toms[idx], self.tractograms[idx], self.beginnings[idx], self.endings[idx], self.means, self.sdevs, self.coord2voxel)

    def __len__(self):
        return len(self.toms)

def OutputToStreamlines(output):
    # This could probably be sped up if using torch operations e.g. https://stackoverflow.com/questions/55757255/replicate-subtensors-in-pytorch
    #seeds, streamlines = output
    streamlines = output

    #seeds = seeds.permute(1, 2, 0) # (3, 32, 32) -> (32, 32, 3)
    #seeds = seeds.cpu().detach().numpy()
    #seeds = np.reshape(seeds, (num_streamlines, 3))

    streamlines = streamlines.permute(1, 2, 0) # (40*3, 32, 32) -> (32, 32, 40*3)
    streamlines = streamlines.cpu().detach().numpy()
    streamlines = np.reshape(streamlines, (num_streamlines, num_points, 3))
    
    #for i in range(len(streamlines)):
    #    streamlines[i] = streamlines[i] + seeds[i]

    return streamlines
    
