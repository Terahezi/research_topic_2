#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:49:40 2019

Texture mapping

@author: hezi
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def initMap(dim=3):
    '''initialize the first texture map 
    with equal probability of being occupied or not
    dimension: 1201x1201
    resolution: 0.05m
    output: initial map as 3D np array'''
    
    # init MAP
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -35  #meters
    MAP['ymin']  = -35
    MAP['xmax']  =  35
    MAP['ymax']  =  35 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey'],dim),dtype=np.float) #DATA TYPE: char or int8
    return MAP['map']

def parseKinect(i,dataset,muBest,disp_stamps,rgb_stamps,encoder_stamps):
    '''transform RGB image and depth image from ir to body and rgb to body
    then transform from body to world using current best particle
    
    given the values d at location X_ind, Y_ind of the disparity image
    obtain the associated color positions in RGB image
    
    output: 1xN numpy arrays
        X_cell: x positions in the grid cell
        Y_cell: y positions in the grid cell
        rgbi_vec: x positions in the RGB image frame
        rgbj_vec: y positions in the RGB image frame
    '''
    # files=os.listdir('/Users/hezi/Downloads/dataRGBD/Disparity%d/'%dataset)
    rgbTime=rgb_stamps[i]
    closestIndex=np.argmin(np.abs(rgbTime-disp_stamps))
    path=os.path.join('/Users/hezi/Downloads/dataRGBD/Disparity%d/'%dataset,\
                      'disparity{}_{}.png'.format(dataset,closestIndex+1))
    img=Image.open(path)
    disparity_img=np.array(img.getdata(),np.uint16)\
    .reshape(img.size[1],img.size[0])
    
    # use vectorized computation
    # vectorize pixel positions in disparity image
    dd=-0.00304*disparity_img+3.31
    Z0=1.03/dd
    i_disparity=np.arange(img.size[1])
    j_disparity=np.arange(img.size[0])
    iDis=np.tile(i_disparity,img.size[0])
    jDis=np.repeat(j_disparity,img.size[1])
    pixels=np.vstack((iDis,jDis,np.ones((img.size[1]*img.size[0])))) # 3x(680x480) array
 
    # compute projection and intrinsics matrices
    # and then get coordinate in optical frame
    cali=np.array([[585.05108211,0,242.94140713],\
                   [0,585.05108211,315.83800193],[0,0,1]])
    Z0=Z0.reshape(1,-1)
    canoProj=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    projIntrin=cali @ canoProj[:,:-1]
    opticalCoordi=np.linalg.inv(projIntrin) @ (pixels*Z0) # 3x(680x480) array
    opticalCoordi=np.vstack((opticalCoordi,np.ones((img.size[1]*img.size[0]))))
    # opticalCoordi: 4x(680x480) array
    
    # compute extrinsics matrix and get coordinates in body frame
    Pbc=np.array([[0.18+0.33276/2],[0.005],[0.36]])  # meters
            
    roll, pitch, yawb=0, 0.36, 0.021   # orientation
    Rz=np.array([[np.cos(yawb),-np.sin(yawb),0],[np.sin(yawb),np.cos(yawb),0],\
                  [0,0,1]])
    Ry=np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],\
                  [-np.sin(pitch),0,np.cos(pitch)]])
    Rx=np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],\
                 [0,np.sin(roll),np.cos(roll)]])
    Rwc=Rz@Ry@Rx
    Roc=np.array([[0,-1,0],[0,0,-1],[1,0,0]])
    R_=Roc@Rwc.T
    p_=-Roc@Rwc.T@Pbc
    extrinInv=np.concatenate((np.concatenate((R_.T,np.array([[0,0,0]]))),\
                              np.concatenate((-R_.T@p_,np.array([[1]])))),axis=1)
    bodyCoordi=extrinInv @ opticalCoordi # 4x(680x480) array
    bodyMask=bodyCoordi[2,:]<-0.100   # height threshold: -0.100m
    bodyCoordiSel=bodyCoordi[:,bodyMask] # 4xNumSel array
    
    # convert from body frame to world frame
    bodyCoordiReduce=np.delete(bodyCoordiSel,2,0) # delete the third row: height
    yaw=muBest[2]
    # Pose of a rigid body, described as a matrix
    # compute the inverse of the pose
    R_pose=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    p_pose=muBest[:-1].reshape(-1,1)
    wRb=np.concatenate((np.concatenate((R_pose,np.array([[0,0]]))),\
                        np.concatenate((p_pose,np.array([[1]])))),axis=1) # 3x3 array
    # R_inv=np.concatenate((np.concatenate((R_pose.T,np.array([[0,0]]))),\
                              #np.concatenate((-R_pose.T@p_pose,np.array([[1]])))),axis=1) # 3x3 array
    # use the invere of pose R to convert back
    particleW=wRb @ bodyCoordiReduce
    xw=particleW[0]         
    yw=particleW[1]
    # yaw_W=yaw                    
    
    # Selection of rgb pixels corresponding to selected pixels in disparity image
    rgbi=(i_disparity.reshape(-1,1)*526.37+dd*(-4.5*1750.46)+19276)/585.051
    rgbj=(j_disparity*526.37+16662)/585.051
    rgbi_vec=rgbi.reshape(1,-1) # 1x(680x480) row vector
    rgbj_vec=np.repeat(rgbj,img.size[1]).reshape(1,-1) # 1x(680x480) row vector
    # rjbPixels=np.vstack((rgbi_vec,rgbj_vec,np.ones((img.size[1]*img.size[0])))) # 3x(680x480) array
    # return valid coordinates in RGB
    rgbi_vec=rgbi_vec.astype(np.int)
    rgbi_vec=rgbi_vec[0,bodyMask]
    rgbj_vec=rgbj_vec.astype(np.int)
    rgbj_vec=rgbj_vec[0,bodyMask]
    
    # convert from world frame to grid cell positions
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -35  #meters
    MAP['ymin']  = -35
    X_cell = np.ceil((xw - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    Y_cell = np.ceil((yw - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    Xmask=np.logical_and(X_cell<1401,X_cell>=0)
    Ymask=np.logical_and(Y_cell<1401,Y_cell>=0)
    cellCoordi=np.vstack((X_cell,Y_cell))
    cellCoordi=cellCoordi[:,np.logical_and(Xmask,Ymask)]
    
    rgbi_vec=rgbi_vec[np.logical_and(Xmask,Ymask)]
    rgbj_vec=rgbj_vec[np.logical_and(Xmask,Ymask)]
            
    return cellCoordi,rgbi_vec,rgbj_vec
            
            
def colorCells(i,dataset,im_texture_mapping,X_cell,Y_cell,rgbi,rgbj):
    '''color the cells in im_texture_mapping using points from RGB image
    that belong to the ground plane'''
    path=os.path.join('/Users/hezi/Downloads/dataRGBD/RGB%d/'%dataset,\
                      'rgb{}_{}.png'.format(dataset,i+1))
    img=plt.imread(path)
    im_texture_mapping[X_cell,Y_cell,:]=img[rgbi,rgbj,:]
    
    return im_texture_mapping
                