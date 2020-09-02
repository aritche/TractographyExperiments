import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import cv2

from dipy.io.streamline import load_trk, save_trk
import nibabel as nib
from nibabel import trackvis
from dipy.tracking import utils
from dipy.io.image import load_nifti 

from plotly import graph_objs as go
import plotly.express as px
from glob import glob

camera = dict(eye=dict(x=0,y=1,z=0))

for tom_fn in glob('../../data/final_hairnet_dataset/not_preprocessed/TOMs/*.nii.gz')[:3]:
    #tom_fn = '../../data/final_hairnet_dataset/preprocessed/TOMs/599469_0_CST_left.nii.gz'
    mean = np.array([-0.002054, 0.001219, -0.008714])
    sdev = np.array([0.047182, 0.038418, 0.087899])

    tom = nib.load(tom_fn).get_data()
    #tom = (tom - mean) / sdev

    xs, ys, zs, us, vs, ws = [], [], [], [], [], []

    i = 0
    for x in range(144):
        for y in range(144):
            for z in range(144):
                u = tom[x][y][z][0]
                v = tom[x][y][z][1]
                w = tom[x][y][z][2]

                if u == 0 and v == 0 and w == 0:
                    continue 

                xs.append(x)
                ys.append(y)
                zs.append(z)

                us.append(tom[x][y][z][0])
                vs.append(tom[x][y][z][1])
                ws.append(tom[x][y][z][2])
                i += 1
                print(i)
                #if i >= 500:
                #    break
            #if i >= 500:
            #    break
        #if i >= 500:
        #    break


    #fig = px.scatter_3d(x=xs, y=ys, z=zs, color=np.array([1 for i in range(len(xs))]), size=[0.5 for item in xs], range_x=[0,144], range_y=[0,144], range_z=[0,144])
    fig = go.Figure(data=go.Cone(x=xs, y=ys, z=zs, u=us, v=vs, w=ws))
    #fig = px.scatter_3d(x=xs, y=ys, z=zs, color=np.array(["rgb(" + str(abs(us[i])) + "," + str(abs(vs[i])) + "," + str(abs(ws[i])) + ")" for i in range(len(xs))]), size=[0.1 for item in xs], range_x=[0,144], range_y=[0,144], range_z=[0,144])
    fig.update_layout(scene_aspectmode='cube',
                      scene=dict(
                        xaxis=dict(range=[0,144]),
                        yaxis=dict(range=[0,144]),
                        zaxis=dict(range=[0,144])
                      ),
                      scene_camera=camera,
                      )
    fig.show()
    """
    print(len(xs), len(ys), len(zs))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, 144])
    ax.set_ylim([0, 144])
    ax.set_zlim([0, 144])
    ax.quiver(xs, ys, zs, us, vs, ws)
    plt.show()
    """

_, affine = load_nifti('../../data/final_hairnet_dataset/preprocessed/TOMs/599469_0_CST_left.nii.gz')
inverse_affine = np.linalg.inv(affine)
for tractogram_fn in glob('../../data/final_hairnet_dataset/not_preprocessed/tractograms/*.trk')[:3]:
    # Load trk
    streamlines, header = trackvis.read(tractogram_fn)
    streamlines = [s[0] for s in streamlines]
    streamlines = np.array(streamlines)

    # Convert to voxel space
    streamlines = utils.apply_affine(aff=inverse_affine, pts=streamlines)

    coords = np.reshape(streamlines, (-1,3))
    xs, ys, zs = coords[:,0], coords[:,1], coords[:,2]

    #fig = px.scatter_3d(x=xs, y=ys, z=zs, color=np.array([xs[i] for i in range(len(xs))]), range_x=[0,144], range_y=[0,144], range_z=[0,144])
    fig = go.Figure(data=go.Scatter3d(x=xs, y=ys, z=zs,mode='markers', marker=dict(size=1, color=[xs[i] for i in range(len(xs))])))
    #fig.update_layout(scene_aspectmode='cube')

    fig.update_layout(scene_aspectmode='cube',
                      scene=dict(
                        xaxis=dict(range=[0,144]),
                        yaxis=dict(range=[0,144]),
                        zaxis=dict(range=[0,144])
                      ),
                      scene_camera=camera
                      )
    fig.show()

    colors = []
    i = 0
    for sl in streamlines:
        colors.extend([i for x in range(len(sl))])
        i += 1
    colors = np.array(colors)
    fig = px.line_3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], color=colors, range_x=[0,144], range_y=[0,144], range_z=[0,144])

    fig.update_layout(scene_aspectmode='cube',
                      scene=dict(
                        xaxis=dict(range=[0,144]),
                        yaxis=dict(range=[0,144]),
                        zaxis=dict(range=[0,144])
                      ),
                      scene_camera=camera
                      )
    fig.show()

    #tractogram = 
    #print(tom.shape)
