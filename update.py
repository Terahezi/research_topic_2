#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:58:06 2019

localization update

@author: hezi
"""
import numpy as np


def parseScan(i,mu,lidar_stamps,lidar_ranges,encoder_stamps):
    '''return laser scan input in world frame
    convert physical dimensions to cell positions
    input:
        i: update sequence (same as predict sequence)
    output:
        lidar scan in world frame as grid cell positions
    '''
    
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    
    # select lidar data whose stamp is closest to encoder's
    encoder_time=encoder_stamps[i]
    
    closest_index=np.argmin(np.abs(encoder_time-lidar_stamps))
    lidar_data=lidar_ranges[:,closest_index]
    
    # take valid indices
    indValid = np.logical_and((lidar_data < 30),(lidar_data> 0.1))
    lidar_data = lidar_data[indValid]
    angles = angles[indValid]
    len_valid=len(lidar_data)   # valid length of range data
    
    # xy position in Lidar frame
    xs0 = lidar_data*np.cos(angles)
    ys0 = lidar_data*np.sin(angles)
    
    # convert from lidar frame to body frame
    p_Lidar_x=0.29833 # meters
    p_Lidar_y=0       # meters
    xb0=xs0+p_Lidar_x # 1x1081 array
    yb0=ys0+p_Lidar_y
    
    # convert from body frame to world frame
    B=np.stack((xb0,yb0,np.ones(len_valid)),axis=-1) # 1081x3 array
    xw0=np.zeros((mu.shape[1],len_valid))       # 100x1081
    yw0=np.zeros((mu.shape[1],len_valid))
    for i in range(mu.shape[1]):
        yaw=mu[2,i]
        # Pose of a rigid body, described as a matrix
        R=np.array([[np.cos(yaw),-np.sin(yaw),mu[0,i]],[np.sin(yaw),np.cos(yaw),mu[1,i]],[0,0,1]])
        particleW=np.dot(R,B.transpose())
        xw0[i]=particleW[0]         
        yw0[i]=particleW[1]
    yaw_W=mu[2]                    # 1x100
    # Y=np.stack((xw0,yw0),axis=-1)
    
    # convert from physical positions to grid cell position
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -35  #meters
    MAP['ymin']  = -35
    MAP['xmax']  =  35
    MAP['ymax']  =  35 
    
    # x-positions of each pixel of the map
    # y-positions of each pixel of the map
    x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) 
    y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res'])
    
    return x_im,y_im,xw0,yw0, yaw_W,len_valid
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    