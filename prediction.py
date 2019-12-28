#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:29:36 2019

localization prediction

@author: hezi
"""
import numpy as np
from scipy import signal

def parseControl(i,encoder_stamps,encoder_counts,imu_stamps,imu_angular_velocity):
    '''return one pair of linear velocity and
    angular velocity at the closest timestamp of
    both encoder and IMU for each encoder data
    input:
        i: predict sequence
    output:
        control: (v,w) 1x2 array
    '''
       
    encoder_time=encoder_stamps[i]
    encoder_data=encoder_counts[:,i]
    
    closest_index1=np.argmin(np.abs(encoder_time-imu_stamps))
    # first apply a low-pass first-order filter with bandwidth around 10Hz
    # then select corresponding imu_data
    b, a = signal.butter(1,0.02,'lowpass')
    imu_angular_velocity = signal.filtfilt(b, a, imu_angular_velocity)
    imu_data=imu_angular_velocity[2,closest_index1]
    
    angular_velocity=imu_data
    vTau=np.sum(encoder_data)/2*0.0011
    Tau=encoder_stamps[i+1]-encoder_time
    wTau=angular_velocity*Tau
    controlTimesT=np.array([vTau,wTau])
    
    return controlTimesT
    
def diff_drive(state,control):
    '''this is the discrete-time 
    differential-drive motion model
    input:
        state=(x,y,theta)   3xN array
        control=(vTau,wTau) Nx2 array
    output:
        newState            3xN array
    '''
    vTau=control[:,0] # 1xN array
    wTau=control[:,1]
    increX=vTau*np.sinc(wTau/2/np.pi)*np.cos(state[2]+wTau/2/np.pi)
    increY=vTau*np.sinc(wTau/2/np.pi)*np.sin(state[2]+wTau/2/np.pi)
    increTheta=wTau
    newState=state+np.array([increX,increY,increTheta])
    
    return newState
    
    
    
    
    
    
    
    
    
    
    
    
    
    