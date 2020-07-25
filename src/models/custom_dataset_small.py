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

num_points = 40
num_streamlines = 64

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
                                                                                                                 # VOLUME SIZE                      # PARAMETERS
        # Encoding (input -> 512 vector)                                                                         # 3 x 144 x 144 x 144 -> 8.9M      (IN * F^3 + 1)*OUT

        # num_points*3, 8, 8
        self.down_conv_1 = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(3,3,3), stride=1, padding=0)     # K   x 136 x 136 x 136 ->        3*(3*9^3+1)     = 6.5K
        self.down_conv_2 = nn.Conv3d(in_channels=3, out_channels=44, kernel_size=(4,4,4), stride=3, padding=0) # K   x 64  x 64  x 64  ->        44*(3*10^3+1)   = 132K
        self.down_conv_3 = nn.Conv3d(in_channels=44, out_channels=32, kernel_size=(3,3,3), stride=2, padding=1)   # K   x 32  x 32  x 32  ->        32*(44*6^3+1)   = 305K
        self.down_conv_4 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3,3,3), stride=3, padding=0)   # K   x 16  x 16  x 16  ->        76*(32*6^3+1)   = 525K


        # num_points*3 x 8 x 8 x 8 
        self.down_conv_5a = nn.Conv3d(in_channels=32, out_channels=num_points*3, kernel_size=(2,2,2), stride=2, padding=0)   # K   x 16  x 16  x 16  ->        76*(32*6^3+1)   = 525K
        self.down_conv_5b = nn.Conv3d(in_channels=32, out_channels=3, kernel_size=(2,2,2), stride=2, padding=0)   # K   x 16  x 16  x 16  ->        76*(32*6^3+1)   = 525K

    def forward(self, x):
        # Encoding
        x = self.down_conv_1(x) # Output: (3, 142, 142, 142)
        x = self.relu(x)
        x = self.down_conv_2(x) # Output: (44, 47, 47, 47)
        x = self.relu(x)
        x = self.down_conv_3(x) # Output: (32, 24, 24, 24)
        x = self.relu(x)
        x = self.down_conv_4(x) # Output: (32, 8, 8, 8)
        x = self.tanh(x)

        p = self.down_conv_5a(x)
        p = p.view(-1, num_points*3, 8, 8)
        s = self.down_conv_5b(x)
        s = s.view(-1, 3, 8, 8)

        return [s, p]

# Custom loss function
def CustomLoss(output, target):
    # Re-implemented MSE loss for efficiency reasons
    seed_output, position_output = output
    seed_target, position_target = target

    position_loss = ((position_output - position_target)**2).mean()
    seed_loss = ((seed_output - seed_target)**2).mean()

    print(position_loss, seed_loss)
    return position_loss + seed_loss
    #return [seed_loss, position_loss]

def get_data(in_fn, out_fn, mean, sdev):
    # Load TOM volume and preprocess
    tom = nib.load(in_fn).get_data() # 144 x 144 x 144 x 3
    tom = (tom - mean) / sdev # normalise based on dataset mean/stdev
    tom = torch.from_numpy(np.float32(tom))
    tom = tom.permute(3, 0, 1, 2) # channels first for pytorch
    
    # Load the tractogram
    streamlines, header = trackvis.read(out_fn)
    streamlines = [s[0] for s in streamlines]
    streamlines = np.array(streamlines)

    # Get seed coordinates and convert streamlines to relative format
    seeds = [sl[0].copy() for sl in streamlines]
    for i in range(len(streamlines)):
        streamlines[i] -= seeds[i]

    # Sort seeds and streamlines by seed points x, then y, then z
    streamlines = list(streamlines)
    streamlines = [x for _, x in sorted(zip(seeds, streamlines), key=lambda pair: [pair[0][0], pair[0][1], pair[0][2]])]
    seeds = sorted(seeds, key=lambda k: [k[0], k[1], k[2]])

    # automatically converts list to numpy array and reshapes it
    # (num_sl, points_per_sl, 3) -> (sqrt(num_sl), sqrt(num_sl), points_per_sl*3)
    # Performed in 2 successive steps because I don't know if it works if I do it in one step
    streamlines = np.reshape(streamlines, (int(num_streamlines**(1/2)), int(num_streamlines**(1/2)), num_points, 3))
    streamlines = np.reshape(streamlines, (int(num_streamlines**(1/2)), int(num_streamlines**(1/2)), num_points*3))
    tractogram = torch.from_numpy(streamlines)
    tractogram = tractogram.permute(2, 0, 1) # channels first for pytorch

    # automatically converts list to numpy array and reshapes it
    #print('Convert to torch...')
    seeds = np.reshape(seeds, (int(num_streamlines**(1/2)), int(num_streamlines**(1/2)), 3))
    seeds = torch.from_numpy(seeds)
    seeds = seeds.permute(2, 0, 1)

    return [tom, [seeds, tractogram]]

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, toms_dir, beginnings_dir, endings_dir, tractograms_dir, means=np.array([]), sdevs=np.array([])):
            
        # Store all input filenames
        self.input_files = glob(toms_dir + '/*.nii.gz')

        # Calculate mean and standard deviation of the input dataset
        if len(means) == 0:
            print("Calculating mean for the dataset...")
            self.means = np.float32(np.array([0, 0, 0]))
            for fn in self.input_files:
                data = nib.load(fn).get_data()
                self.means += np.mean(data.reshape((-1,3)), axis=0)
            self.means = self.means/(len(self.input_files))

            print("Calculating sdev for the dataset...")
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
    seeds, streamlines = output

    seeds = seeds.permute(1, 2, 0) # (3, 32, 32) -> (32, 32, 3)
    seeds = seeds.cpu().detach().numpy()
    seeds = np.reshape(seeds, (num_streamlines, 3))

    streamlines = streamlines.permute(1, 2, 0) # (40*3, 32, 32) -> (32, 32, 40*3)
    streamlines = streamlines.cpu().detach().numpy()
    streamlines = np.reshape(streamlines, (num_streamlines, num_points, 3))
    
    for i in range(len(streamlines)):
        streamlines[i] = streamlines[i] + seeds[i]

    return streamlines
    
