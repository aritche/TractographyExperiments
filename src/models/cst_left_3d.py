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

        """                                                                                                 # 3   x 256 x 256 -> 196K               
        self.down_conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=2, padding=3)    # 32  x 128 x 128 -> 524K               (8*8*3+1)*32 = 6K
        self.down_conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=8, stride=2, padding=3)   # 64  x 64  x 64  -> 262K               (8*8*32+1)*64 = 131K
        self.down_conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=6, stride=2, padding=2)  # 128 x 32  x 32  -> 131K               
        self.down_conv_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1) # 256 x 16  x 16  -> 65K                
        self.down_conv_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1) # 256 x 16  x 16  -> 65K                
        self.down_conv_6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1) # 512 x 8   x 8   -> 32K                
        self.down_conv_7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1) # 512 x 8   x 8   -> 32K                
        self.down_pool_8 = nn.MaxPool2d(kernel_size=8)                                                      # 512             -> 0.5K               
        """

        # Decoding (512 vector -> 32 x 32 x 512 volume)
        self.up_linear_1 = nn.Linear(in_features=512, out_features=1024)
        self.up_linear_2 = nn.Linear(in_features=1024, out_features=4096)
        self.up_conv_3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.up_conv_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.up_conv_5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        # Position decoder (32 x 32 x 512 volume -> 32 x 32 x 300)
        self.pos_conv_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.pos_conv_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.pos_conv_3 = nn.Conv2d(in_channels=512, out_channels=40*3, kernel_size=1, stride=1, padding=0)

        # Curvature decoder (32 x 32 x 512 volume -> 32 x 32 x 100)
        self.curv_conv_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.curv_conv_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.curv_conv_3 = nn.Conv2d(in_channels=512, out_channels=100, kernel_size=1, stride=1, padding=0)

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

        x = x.view(-1, 512)

        x = self.up_linear_1(x)
        x = self.relu(x)
        x = self.up_linear_2(x)
        x = self.relu(x)

        x = x.view(-1, 1, 64, 64)
        x = self.upsample(x)

        x = x.view(-1, 256, 8, 8)
        x = self.up_conv_3(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.up_conv_4(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.up_conv_5(x)
        x = self.relu(x)

        #c = self.curv_conv_1(x)
        #c = self.relu(c)
        #c = self.curv_conv_2(c)
        #c = self.tanh(c)
        #c = self.curv_conv_3(c)

        p = self.pos_conv_1(x)
        p = self.relu(p)
        p = self.pos_conv_2(p)
        p = self.tanh(p)
        p = self.pos_conv_3(p)

        #return [c, p]
        return p

# Custom loss function
def CustomLoss(output, target):
    #curvature_output, position_output = output
    #curvature_target, position_target = target

    criterion = nn.MSELoss()
    position_loss = criterion(output, target)

    #position_loss  = criterion(curvature_output, curvature_target)
    #curvature_loss = criterion(position_output, position_target)

    #return position_loss + curvature_loss
    return position_loss

#def get_data(in_fn, ends_fn, out_fn):
def get_data(in_fn, out_fn, mean, sdev):
    do_flip_X = False if random.randint(0,1) == 0 else True
    do_flip_Y = False if random.randint(0,1) == 0 else True
    do_flip_Z = False if random.randint(0,1) == 0 else True

    # Load volume
    tom = nib.load(in_fn).get_data() # 144 x 144 x 144 x 3
    #tom = np.sum(tom, axis=0)
    #tom = cv2.resize(tom, (256, 256))

    # Preprocess input
    tom = (tom - mean) / sdev # normalise based on dataset mean/stdev
    if do_flip_X:
        tom = tom[::-1,:,:]
    if do_flip_Y:
        tom = tom[:,::-1,:]
    if do_flip_Z:
        tom = tom[:,:,::-1]
    tom = torch.from_numpy(np.float32(tom))
    tom = tom.permute(3, 0, 1, 2) # channels first for pytorch
    
    # Load the tractogram
    tractogram = load_trk(out_fn, 'same', bbox_valid_check=False)
    streamlines = tractogram.streamlines

    # Preprocess the streamlines
    num_points = 40
    num_streamlines = 1024

    streamlines = select_random_set_of_streamlines(streamlines, num_streamlines)
    streamlines = set_number_of_points(streamlines, num_points)
    streamlines = np.array(streamlines)
    if len(streamlines) < num_streamlines:
        temp_streamlines = np.zeros((num_streamlines, num_points, 3))
        temp_streamlines[:streamlines.shape[0],:streamlines.shape[1], :streamlines.shape[2]] = streamlines
        streamlines = np.float32(temp_streamlines)
    streamlines = np.reshape(streamlines, (int(num_streamlines**(1/2)), int(num_streamlines**(1/2)), num_points*3))

    #tractogram = (tractogram - np.min(tractogram)) / (np.max(tractogram) - np.min(tractogram))
    tractogram = torch.from_numpy(streamlines)
    tractogram = tractogram.permute(2, 0, 1) # channels first for pytorch

    return [tom, tractogram]

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, toms_dir, endings_dir, tractograms_dir):
            
        # Only allow CST_left files
        all_TOMs = glob(toms_dir + '/*.nii.gz')
        self.input_files = []
        for fn in all_TOMs:
            if 'CST_left.nii.gz' in fn: 
                self.input_files.append(fn)

        self.means = np.float32(np.array([0, 0, 0]))
        for fn in self.input_files:
            data = nib.load(fn).get_data()
            self.means += np.mean(data.reshape((-1,3)), axis=0)
        self.means = self.means/(len(self.input_files))

        print("NORMALISING WITH MEANS: %f, %f, %f" % (self.means[0], self.means[1], self.means[2]))

        self.sdevs = np.float32(np.array([0, 0, 0]))
        squared_diffs_sum = np.float32(np.array([0,0,0]))
        for fn in self.input_files:
            data = nib.load(fn).get_data()
            pixels = data.reshape((-1,3))
            squared_diffs_sum += np.sum((pixels - self.means)**2, axis=0)
        self.sdevs  = (squared_diffs_sum / (len(self.input_files) * 144*144*144))**(1/2)

        print("NORMALISING WITH SDEVS: %f, %f, %f" % (self.sdevs[0], self.sdevs[1], self.sdevs[2]))

        all_tractograms = glob(tractograms_dir + '/*.trk')
        self.output_files = []
        for fn in all_tractograms:
            if 'CST_left.trk' in fn:
                self.output_files.append(fn)

        # Sort for consistency
        self.input_files.sort()
        self.output_files.sort()


    # Given an index, return the loaded [data, label]
    def __getitem__(self, idx):
        #return get_data(self.input_files[idx], self.endings_files[idx], self.output_files[idx])
        return get_data(self.input_files[idx], self.output_files[idx], self.means, self.sdevs)

    def __len__(self):
        return len(self.input_files)
