#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:27:03 2021

@author: benjamin
"""



def create_tst_pertubations_mm(array, moving_mean_size = 60):
    # 
    """
    Subtracts a temporal moving mean of each pixel in array (Reynolds averaging). See Christen 2012 for details.    
    
    Parameters
    ----------
    array: numpy 3D array
        3D array of ideally - stabilized brightness temperature video from an overhead drone flight or a corrected oblique pole acquisition.
        x,y is the image
        z is the time dimension
    moving_mean_size: int
        The size of the moving mean to be subtracted in time.
    
    
    
    """

    resultarr = np.zeros(np.shape(array))
    bar = progressbar.ProgressBar(maxval=len(array), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]) 
    bar.start()
    bar_iterator = 0
    for i in range(0,len(array)):
        # moving mean array = actarray:
        if i == 0:
            actarray = array[0:moving_mean_size*2+1]
        elif i != 0 and i != len(array) and i-(moving_mean_size)>= 0 and i+(moving_mean_size)<= len(array)-1:
            actarray = array[int(i-moving_mean_size):int(i+moving_mean_size)+1]
        elif i-(moving_mean_size)<= 0:
            actarray = array[0:moving_mean_size*2+1]   
        elif i+(moving_mean_size)>= len(array):
            actarray = array[len(array)-(2*moving_mean_size)-1:len(array)]        
        if i == len(array)-1:
            actarray = array[len(array)-(2*moving_mean_size)-1:len(array)]
        
        resultarr[i] = array[i]-np.mean(actarray, axis=0)
        bar.update(bar_iterator+1)
        bar_iterator += 1
                
    bar.finish()
    return(resultarr)





def create_tst_pertubations_spmm(array, moving_mean_size = 60):
        """
    Subtracts a spatiotemporal moving mean around each layer in array (Reynolds averaging). See Christen 2012 for details.    
    
    Parameters
    ----------
    array: numpy 3D array
        3D array of ideally - stabilized brightness temperature video from an overhead drone flight or a corrected oblique pole acquisition.
        x,y is the image
        z is the time dimension
    moving_mean_size: int
        The size of the moving mean to be subtracted in time.
    
    
    
    """
    resultarr = np.zeros(np.shape(array))
    bar = progressbar.ProgressBar(maxval=len(array), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]) 
    bar.start()
    bar_iterator = 0
    arr_spmean = np.mean(array, axis=(1,2))
    arr_spmean = arr_spmean[:,np.newaxis,np.newaxis]
    arr_spperturb = np.ones(array.shape, dtype="int")
    arr_spperturb = arr_spperturb*arr_spmean
    array = array-arr_spperturb 
    for i in range(0,len(array)):
        # moving mean array = actarray:
        if i == 0:
            actarray = array[0:moving_mean_size*2+1]
        elif i != 0 and i != len(array) and i-(moving_mean_size)>= 0 and i+(moving_mean_size)<= len(array)-1:
            actarray = array[int(i-moving_mean_size):int(i+moving_mean_size)+1]
        elif i-(moving_mean_size)<= 0:
            actarray = array[0:moving_mean_size*2+1]   
        elif i+(moving_mean_size)>= len(array):
            actarray = array[len(array)-(2*moving_mean_size)-1:len(array)]        
        if i == len(array)-1:
            actarray = array[len(array)-(2*moving_mean_size)-1:len(array)]
        
        resultarr[i] = array[i]-np.mean(actarray, axis=0)
        bar.update(bar_iterator+1)
        bar_iterator += 1
                
    bar.finish()
    return(resultarr)

