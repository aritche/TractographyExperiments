import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import sys
import random

from models.relative_sorted import CustomDataset, OutputToStreamlines
model_name = "relative_sorted"
epoch_number = "250"
test_dir = '../data/test_pre_sampled'
means = np.array([-0.000018, 0.000004, -0.000012])
sdevs = np.array([0.048257, 0.039410, 0.088247])
fn = './results/' + model_name + '/epoch_' + epoch_number + '.pth'

def normalise_stack_generic(im_x, im_y, im_z, min_x, max_x, min_y, max_y, min_z, max_z):
    im_x = (im_x - min_x) / (max_x - min_x)*255
    im_y = (im_y - min_y) / (max_y - min_y)*255
    im_z = (im_z - min_z) / (max_z - min_z)*255
    
    return np.stack([im_x, im_y, im_z], axis=2)

def plot_input_volume(input_data):
    min_x, min_y, min_z = np.min(input_data[:,:,:,0]), np.min(input_data[:,:,:,1]), np.min(input_data[:,:,:,2])
    max_x, max_y, max_z = np.max(input_data[:,:,:,0]), np.max(input_data[:,:,:,1]), np.max(input_data[:,:,:,2])

    
    cv2.namedWindow('input_data_x', cv2.WINDOW_NORMAL)
    cv2.namedWindow('input_data_y', cv2.WINDOW_NORMAL)
    cv2.namedWindow('input_data_z', cv2.WINDOW_NORMAL)

    i = 60
    #for i in range(len(input_data[2])):
    input_slice = input_data[:,:,i,:]
    input_norm = normalise_stack_generic(input_slice[:,:,0], input_slice[:,:,1], input_slice[:,:,2], min_x, max_x, min_y, max_y, min_z, max_z)
    cv2.imshow('input_data_z', np.uint8(input_norm))

    input_slice = input_data[:,i,:,:]
    input_norm = normalise_stack_generic(input_slice[:,:,0], input_slice[:,:,1], input_slice[:,:,2], min_x, max_x, min_y, max_y, min_z, max_z)
    cv2.imshow('input_data_y', np.uint8(input_norm))

    input_slice = input_data[i,:,:,:]
    input_norm = normalise_stack_generic(input_slice[:,:,0], input_slice[:,:,1], input_slice[:,:,2], min_x, max_x, min_y, max_y, min_z, max_z)
    cv2.imshow('input_data_x', np.uint8(input_norm))
    cv2.waitKey(0)
    

def plot_output_volume(output, label):
    seeds, coords = output
    label_seeds, label_coords = label

    """
    PLOT THE SEEDS
    """
    seeds = seeds.permute(1, 2, 0) # (3, 32, 32) -> (32, 32, 3)
    seeds = seeds.cpu().detach().numpy()
    label_seeds = label_seeds.permute(1, 2, 0) # (3, 32, 32) -> (32, 32, 3)
    label_seeds = label_seeds.cpu().detach().numpy()

    min_x = np.min(seeds[:,:,0]) if np.min(seeds[:,:,0]) < np.min(label_seeds[:,:,0]) else np.min(label_seeds[:,:,0])
    min_y = np.min(seeds[:,:,1]) if np.min(seeds[:,:,1]) < np.min(label_seeds[:,:,1]) else np.min(label_seeds[:,:,1])
    min_z = np.min(seeds[:,:,2]) if np.min(seeds[:,:,2]) < np.min(label_seeds[:,:,2]) else np.min(label_seeds[:,:,2])
    max_x = np.max(seeds[:,:,0]) if np.max(seeds[:,:,0]) < np.max(label_seeds[:,:,0]) else np.max(label_seeds[:,:,0])
    max_y = np.max(seeds[:,:,1]) if np.max(seeds[:,:,1]) < np.max(label_seeds[:,:,1]) else np.max(label_seeds[:,:,1])
    max_z = np.max(seeds[:,:,2]) if np.max(seeds[:,:,2]) < np.max(label_seeds[:,:,2]) else np.max(label_seeds[:,:,2])

    #seeds_norm = normalise_stack_2d(seeds[:,:,0], seeds[:,:,1], seeds[:,:,2])
    seeds_norm = normalise_stack_generic(seeds[:,:,0], seeds[:,:,1], seeds[:,:,2], min_x, max_x, min_y, max_y, min_z, max_z)
    cv2.namedWindow('seeds_norm', cv2.WINDOW_NORMAL)

    #label_seeds_norm = normalise_stack_2d(label_seeds[:,:,0], label_seeds[:,:,1], label_seeds[:,:,2])
    label_seeds_norm = normalise_stack_generic(label_seeds[:,:,0], label_seeds[:,:,1], label_seeds[:,:,2], min_x, max_x, min_y, max_y, min_z, max_z)
    cv2.namedWindow('label_seeds_norm', cv2.WINDOW_NORMAL)

    cv2.imshow('seeds_norm', np.uint8(seeds_norm))
    cv2.imshow('label_seeds_norm', np.uint8(label_seeds_norm))
    cv2.waitKey(1)


for epoch in ["0", "10", "50", "100", "250"]:
    fn = './results/' + model_name + '/epoch_' + epoch + '.pth'

    print("Doing model %s..." % (epoch))
    model = torch.load(fn)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = CustomDataset(test_dir + '/CST_TOMs', test_dir + '/CST_beginnings_masks', test_dir + '/CST_endings_masks', test_dir + '/CST_tractograms', means=means, sdevs=sdevs) 
    testloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    torch.cuda.empty_cache()

    i = 0
    cv2.namedWindow('encoding', cv2.WINDOW_NORMAL)
    cv2.namedWindow('encoding_normal', cv2.WINDOW_NORMAL)

    #for inputs, labels in testloader:
    for i in range(7):
        print("Example " + str(i))
        # Pass to the model
        #inputs, labels = inputs.to(device), [labels[0].to(device), labels[1].to(device)]
        #outputs = model.forward(inputs)

        #input_data = inputs[0]
        #input_data = input_data.permute(1,2,3,0)
        #input_data = input_data.cpu().detach().numpy()
        #plot_input_volume(input_data)

        #print(input_data.size())

        """
        x = model.down_conv_1(inputs)
        x = model.relu(x)
        x = model.down_conv_2(x)
        x = model.relu(x)
        x = model.down_conv_3(x)
        x = model.relu(x)
        x = model.down_conv_4(x)
        x = model.relu(x)
        x = model.down_conv_5(x)
        x = model.relu(x)
        x = model.down_conv_6(x)
        x = model.relu(x)
        x = model.down_conv_7(x)
        x = model.relu(x)
        x = model.down_pool_8(x)
        x = model.tanh(x)

        x = x.view(-1, 512)
        """

        if i == 6:
            latent = torch.ones(1,512) * -1
        if i == 5:
            latent = torch.zeros(1,512)
        if i == 4:
            latent = torch.ones(1,512)
        if i == 3:
            latent = np.random.rand(1, 512)
            latent[latent > 0.9] = 0
            latent[latent > 0] = 1
            latent = torch.from_numpy(latent)
        if i == 2:
            latent = np.random.rand(1, 512)
            latent[latent > 0.9] = 1
            latent[latent < 1] = 0
            latent = torch.from_numpy(latent)
        if i == 1:
            # Random in range [-1,1]
            latent = torch.rand(1, 512) * (-1 - 1) + 1

        if i == 0:
            # Random in range [0,1]
            latent = torch.rand(1,512)
        x = latent.float().to(device)
        #x = x.view(-1, 1, 16, 32)

        #x = torch.nn.functional.interpolate(x, size=(32, 64))
        #print(x.size())

        # Visualise the latent representation
        vector = x.cpu().detach().numpy()
        vector = np.reshape(vector, (16, 32))

        # Normalise based on the fact that tanh(x) maps to [-1, 1]
        vector = ((vector + 1) / (2)) * 255
        #vector = (vector - np.min(vector)) / (np.max(vector) - np.min(vector)) * 255
        #vector = cv2.resize(vector, (320, 160), cv2.INTER_NEAREST)
        #cv2.imwrite('example_' + str(i) + '_' + epoch + '.bmp', np.uint8(vector))
        cv2.imshow('encoding_normal', np.uint8(vector))
        #cv2.waitKey(0)

        #batch_size = x.size()[0]
        #latent += torch.rand() / 100
        #print(latent)
        #x = latent.copy()
        #x = torch.zeros(batch_size, 512).to(device)
        #x = torch.zeros(1, 512)
        #######

        print(x.size())
        x = model.up_linear_1(x)
        x = model.relu(x)
        x = model.up_linear_2(x)
        x = model.relu(x)

        x = x.view(-1, 1, 64, 64)
        x = model.upsample(x)

        x = x.view(-1, 256, 8, 8)
        x = model.up_conv_3(x)
        x = model.relu(x)
        x = model.upsample(x)
        x = model.up_conv_4(x)

        x = model.relu(x)
        x = model.upsample(x)
        x = model.up_conv_5(x)
        x = model.relu(x)
        #x = torch.nn.functional.interpolate(x, size=(512, 512))
        #print(x.size())


        s = model.seed_conv_1(x)
        s = model.relu(s)
        s = model.seed_conv_2(s)
        s = model.tanh(s)
        s = model.seed_conv_3(s)

        p = model.pos_conv_1(x)
        p = model.relu(p)
        p = model.pos_conv_2(p)
        p = model.tanh(p)
        p = model.pos_conv_3(p)

        p, s = [p[0], s[0]]

        seeds = s.permute(1, 2, 0) # (3, 32, 32) -> (32, 32, 3)
        seeds = seeds.cpu().detach().numpy()
        r, g, b = seeds[:,:,0], seeds[:,:,1], seeds[:,:,2]
        r = (r - np.min(r)) / (np.max(r) - np.min(r))*255
        g = (g - np.min(g)) / (np.max(g) - np.min(g))*255
        b = (b - np.min(b)) / (np.max(b) - np.min(b))*255
        result = np.stack([r, g, b], axis=2)
        cv2.namedWindow('SEEDings', cv2.WINDOW_NORMAL)
        cv2.imshow('SEEDings', np.uint8(result))
        cv2.waitKey(0)

        #plot_output_volume([s, p], [labels[0][0], labels[1][0]])
        """
        """

        # Remove the batching to get the single item
        #input = inputs[0]
        #label = [labels[0][0], labels[1][0]]
        #output = [outputs[0][0], outputs[1][0]]

        i += 1
