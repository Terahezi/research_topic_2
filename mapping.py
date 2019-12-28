#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 21:21:00 2019

mapping from the first laser scan and plot the map

@author: hezi
"""

import numpy as np

def initMap():
    '''initialize the first log-odds map 
    with equal probability of being occupied or not
    dimension: 1201x1201
    resolution: 0.05m
    output: initial map as np array'''
    
    # init MAP
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -35  #meters
    MAP['ymin']  = -35
    MAP['xmax']  =  35
    MAP['ymax']  =  35 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float) #DATA TYPE: char or int8
    return MAP['map']


def Lidar_wTbbTl(ranges):
    '''current version: just to transform the first Lidar
    scan so as to obtain mt
    final version: transform Laser scan zt to world frame
    input: 
    output: coordinates of ranges (Lidar scan) in world frame'''
    
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    # ranges = np.load("test_ranges.npy")
    
    # take valid indices
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    
    # xy position in the sensor frame
    xs0 = ranges*np.cos(angles)
    ys0 = ranges*np.sin(angles)
    
    # convert from lidar frame to body frame
    p_Lidar_x=0.29833 # meters
    p_Lidar_y=0       # meters
    xb0=xs0+p_Lidar_x
    yb0=ys0+p_Lidar_y
    
    # convert from body frame to world frame
    xw0=xb0
    yw0=yb0
    Y=np.stack((xw0,yw0),axis=-1)
    
    return Y
    
def formCells(Y,xInit,yInit):
    '''convert from meters to cells for passing to bresenham2D
    input:
        Y: Lidar scan in world frame
    output:
        sx,sy: start point of ray in grid
        ex,ey: end point of ray in grid
    '''
    
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -35  #meters
    MAP['ymin']  = -35
    
    sx=np.ceil((xInit - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    sy=np.ceil((yInit - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
    xis = np.ceil((Y[:,0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((Y[:,1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    return sx, sy, xis, yis
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    