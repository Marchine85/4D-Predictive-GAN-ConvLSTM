"""
code from: https://github.com/meetps/tf-3dgan/tree/master
"""

import sys
import os

import scipy.ndimage as nd
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as sk

from mpl_toolkits import mplot3d


import trimesh
from stl import mesh


def getVF(path):
    raw_data = tuple(open(path, 'r'))
    header = raw_data[1].split()
    n_vertices = int(header[0])
    n_faces = int(header[1])
    vertices = np.asarray([map(float,raw_data[i+2].split()) for i in range(n_vertices)])
    faces = np.asarray([map(int,raw_data[i+2+n_vertices].split()) for i in range(n_faces)]) 
    return vertices, faces

def plotFromVF(vertices, faces, obj):
    input_vec = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            input_vec.vectors[i][j] = vertices[f[j],:]
    '''
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(input_vec.vectors))
    scale = input_vec.points.flatten('C')
    axes.auto_scale_xyz(scale, scale, scale)
    plt.show()
    '''
    input_vec.save(obj+'.stl')

def plotFromVoxels(voxels, i=None):
    #print(voxels.nonzero().shape)
    z,x,y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', c= 'black', marker='s')
    #ax.voxels(voxels, facecolors="black", edgecolor='k')
    ax.set_xlim(0, voxels.shape[0])
    ax.set_ylim(0, voxels.shape[1])
    ax.set_zlim(0, voxels.shape[2])
    if i is not None:
        ax.set_title(i)
    plt.show()

def getVFByMarchingCubes(voxels):
    v, f, normals, values =  sk.marching_cubes(voxels,
                                               level=0.01, 
                                               spacing=(1, 1, 1), 
                                               gradient_direction='descent', 
                                               step_size=1, 
                                               allow_degenerate=False, 
                                               method='lewiner', mask=None)
    return v, f

def plotMeshFromVoxels(voxels, obj='toilet'):
    v,f = getVFByMarchingCubes(voxels)
    plotFromVF(v,f,obj)

def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))

def plotFromVertices(vertices):
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    axes.scatter(vertices.T[0,:],vertices.T[1,:],vertices.T[2,:])
    plt.show()

def getVolumeFromOFF(path, sideLen=32):
    mesh = trimesh.load(path)
    volume = trimesh.voxel.Voxel(mesh, 0.5).raw
    (x, y, z) = map(float, volume.shape)
    volume = nd.zoom(volume.astype(float), 
                     (sideLen/x, sideLen/y, sideLen/z),
                     order=1, 
                     mode='nearest')
    volume[np.nonzero(volume)] = 1.0
    return volume.astype(np.bool)

def getVoxelFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels,(1,1),'constant',constant_values=(0,0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2,2,2), mode='constant', order=0)
    if cube_len != 32 and cube_len == 128:
        voxels = nd.zoom(voxels, (4,4,4), mode='constant', order=0)
    return voxels

