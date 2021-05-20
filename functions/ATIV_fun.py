#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:27:03 2021

@author: benjamin
"""
import numpy as np
from functions.openpiv_fun import *
from collections import Counter
from statistics import mode 
from PyEMD.EEMD import *
import math
from scipy.signal import hilbert
import copy
import gc
import progressbar


def fill_weight(arr_lst, time_lst):
    arr = np.stack(arr_lst)
    w = np.array(time_lst)/10
    arr_w = np.average(arr, axis = 0, weights=w)
    return(arr_w)



def create_tst_perturbations_mm(array, moving_mean_size = 60): 
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





def create_tst_perturbations_spmm(array, moving_mean_size = 60):
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





def find_interval(signal, fs, imf_no = 1):
    """
    Compute the interval setting for the TIV. 
    Calculating a hilbert transform from the instrinct mode functions (imfs).
    This is based on the hilbert-huang transform assuming non-stationarity of the given dataset.
    The function returns a suggested interval (most powerful frequency) based on the weighted average. 
    The weights are the calculated instantaneous energies. 
    
    Parameters
    ----------
    signal: 1d np.ndarray
        a one dimensional array which contains the brightness temperature (perturbation) of one pixel over time.
    fs: int 
        the fps which was used to record the imagery 
    imf_no: int (default 1)
        The imf which will be used to calculate the interval on. IMF 1 is the one with the highest frequency.
    Returns
    -------
    recommended_interval : float
        The found most powerful interval in float. Needs rounding to the next int.
    
    """
    eemd = EEMD()
    imfs = eemd.eemd(signal)
    imf = imfs[imf_no-1,:]        
    
    
    sig = hilbert(imf)
    
    energy = np.square(np.abs(sig))
    phase = np.arctan2(sig.imag, sig.real)
    omega = np.gradient(np.unwrap(phase))
    
    omega = fs/(2*math.pi)*omega
    #omegaIdx = np.floor((omega-F[0])/FResol)+1;
    #freqIdx = 1/omegaIdx;
    
    insf = omega
    inse = energy
    
    rel_inse = inse/np.nanmax(inse)
    insf_weigthed_mean = np.average(insf,weights=rel_inse)
    insp = 1/insf_weigthed_mean
    recommended_interval = np.round(fs*insp,1)

    gc.collect()
    return(recommended_interval)
    
    
    
def randomize_find_interval (data,  rec_freq = 1, plot_hht = False, outpath = "/", figname = "hht_fig"):
    """
    Compute the interval setting for the TIV. 
    Basically a wrapper function for the find_interval function
    
    
    Parameters
    ----------
    data: 3d np.ndarray
        a three dimensional array which contains the brightness temperature (perturbation).
    rec_freq: int (default 1)
        the fps which was used to record the imagery 
    plot_hht: boolean (default False) (not implemented yet!)
        Boolean flag to plot the results for review (not implemented yet!)
    outpath: string (default "/") (not implemented yet!)
        The outpath for the plots - only the last plot of 10 plots is saved in this directory
        Set to a proper directory when used with Boolean flag. (not implemented yet!)
    figname: string (default "hht_fig") (not implemented yet!)
        The output figure name. (not implemented yet!)
    Returns
    -------
    [mode(interval_lst),interval_lst] : list
        The found most occuring and powerful period, and the list which was used to calculate this
    
    """
    masked_boo = True     
    for i in range(0,11):
        while masked_boo:
            rand_x = np.round(np.random.rand(),2)
            rand_y = np.round(np.random.rand(),2)
            
            x = np.round(50+(225-50)*rand_x)
            y = np.round(50+((225-50)*rand_y))
            
            if plot_hht:
                print(x)
                print(y)
            
            
        
        #print(data.shape)
            pixel = data[:,int(x),int(y)]
            
            if np.isnan(np.sum(pixel)):
                masked_boo = True
            else:
                masked_boo = False
                
                
            
        #print(data.shape)
            
        
        act_interval1 = find_interval(pixel, rec_freq, imf_no = 1)
        act_interval2 = find_interval(pixel, rec_freq, imf_no = 2)
        
        act_intervals = [round(act_interval1),round(act_interval2)]
        act_intervals2 = [act_interval1,act_interval2]
        
        
        if i == 0:
            interval_lst = copy.copy([act_intervals])
            interval_lst2 = copy.copy([act_intervals2])
        else:
            interval_lst.append(act_intervals) 
            interval_lst2.append(act_intervals2)
            #np.append(interval_lst,act_interval,0)
    
    try:
        first_most = mode(list(zip(*interval_lst))[0])
    except:
        d_same_count_intervals = Counter(list(zip(*interval_lst))[0])
        d_same_count_occ = Counter(d_same_count_intervals.values())
        for value in d_same_count_occ.values():
            if value == 2:
                first_most = np.mean(list(d_same_count_intervals.keys())[0:2])
            if value == 3:
                first_most = np.mean(list(d_same_count_intervals.keys())[0:3])
            if value == 4:
                first_most = np.mean(list(d_same_count_intervals.keys())[0:4])
            if value == 5:
                first_most = np.mean(list(d_same_count_intervals.keys())[0:5])   
    try:
        second_most = mode(list(zip(*interval_lst))[1]) 
    except:
        sec_most_lst = list(zip(*interval_lst))[1]
        d_same_count_intervals = Counter(list(zip(*interval_lst))[1])
        # 
        try:
            if first_most in d_same_count_intervals.keys():
                sec_most_lst = np.delete(sec_most_lst, np.where(sec_most_lst == first_most))       
                second_most = mode(sec_most_lst)
            else:
                raise(ValueError())
        except:
            d_same_count_occ = Counter(d_same_count_intervals.values())
            for value in d_same_count_occ.values():
                if value == 2:
                    second_most = np.mean(list(d_same_count_intervals.keys())[0:2])
                if value == 3:
                    second_most = np.mean(list(d_same_count_intervals.keys())[0:3])
                if value == 4:
                    second_most = np.mean(list(d_same_count_intervals.keys())[0:4])
                if value == 5:
                    second_most = np.mean(list(d_same_count_intervals.keys())[0:5])            
 
    return([first_most, second_most, interval_lst2])


def window_correlation_tiv(frame_a, frame_b, window_size, overlap_window, overlap_search_area, corr_method, search_area_size_x, 
                           search_area_size_y=0,  window_size_y=0, mean_analysis = True, std_analysis = True, std_threshold = 10):
   
    
   
    if not (window_size-((search_area_size_x-window_size)/2))<= overlap_search_area:
        raise ValueError('Overlap or SearchArea has to be bigger: ws-(sa-ws)/2)<=ol')
        
     
    
    n_rows, n_cols = get_field_shape(frame_a.shape, search_area_size_x, overlap_search_area )   
    u = np.zeros((n_rows, n_cols))
    v = np.zeros((n_rows, n_cols))   

    for k in range(n_rows):

        for m in range(n_cols):
            # Select first the largest window, work like usual from the top left corner
            # the left edge goes as: 
            # e.g. 0, (search_area_size - overlap), 2*(search_area_size - overlap),....
            il = k*(search_area_size_x - overlap_search_area)#k*(search_area_size_x - (search_area_size_x-1)) #
            ir = il + search_area_size_x
            
            # same for top-bottom
            jt = m*(search_area_size_x - overlap_search_area)#m*(search_area_size_x - (search_area_size_x-1)) #
            jb = jt + search_area_size_x
            
            # pick up the window in the second image
            window_b = frame_b[il:ir, jt:jb]            
            #plt.imshow(window_b)
            #plt.imshow(frame_b)
            #window_a_test = frame_a[il:ir, jt:jb]    
          
            rolling_wind_arr = moving_window_array(window_b, window_size, overlap_window)
            
            
            # now shift the left corner of the smaller window inside the larger one
            il += (search_area_size_x - window_size)//2
            # and it's right side is just a window_size apart
            ir = il + window_size
            # same same
            jt += (search_area_size_x - window_size)//2
            jb =  jt + window_size
        
            window_a = frame_a[il:ir, jt:jb]
            
            
            rep_window_a = np.repeat(window_a[ :, :,np.newaxis], rolling_wind_arr.shape[0], axis=2)
            rep_window_a = np.rollaxis(rep_window_a,2)
            

            
            

            
            
            if corr_method == "greyscale": 
            
                dif = rep_window_a - rolling_wind_arr
                dif_sum = np.sum(abs(dif),(1,2))
                shap = int(np.sqrt( rep_window_a.shape[0]))
                dif_sum_reshaped = np.reshape(dif_sum, (shap,shap))
                dif_sum_reshaped = (dif_sum_reshaped*-1)+np.max(dif_sum_reshaped)
                row, col = find_subpixel_peak_position(corr=dif_sum_reshaped)            
                row =  row -((shap-1)/2)
                col =  col - ((shap-1)/2)
                

                
                
                if mean_analysis:
                    if np.all(window_a==np.mean(window_a)): 
                        col = np.nan
                        row = np.nan
                
                if std_analysis:
                    if np.std(window_a)< std_threshold:
                        col = np.nan
                        row = np.nan
                        
                
                
                u[k,m],v[k,m] = col, row
      
            
            if corr_method == "rmse":         
                rmse = np.sqrt(np.mean((rolling_wind_arr-rep_window_a)**2,(1,2)))
      
                shap = int(np.sqrt( rep_window_a.shape[0]))
                rmse_reshaped = np.reshape(rmse, (shap,shap))
                rmse_reshaped = (rmse_reshaped*-1)+np.max(rmse_reshaped)
               
                
                row, col = find_subpixel_peak_position(rmse_reshaped)
                row =  row -((shap-1)/2)
                col =  col - ((shap-1)/2)
                
                 
                if mean_analysis:
                    if np.all(window_a==np.mean(window_a)):
                        col = np.nan
                        row = np.nan
                
                if std_analysis:
                    if np.std(window_a)< std_threshold:
                        col = np.nan
                        row = np.nan
                

                u[k,m],v[k,m] = col, row
                

            if corr_method == "ssim":
                ssim_lst = ssim(rolling_wind_arr,rep_window_a)
                shap = int(np.sqrt( rep_window_a.shape[0]))
                dif_sum_reshaped = np.reshape(ssim_lst, (shap,shap))
                
                row, col = find_subpixel_peak_position(dif_sum_reshaped)
                row =  row -((shap-1)/2)
                col =  col - ((shap-1)/2)
                
                if mean_analysis:
                    if np.all(window_a==np.mean(window_a)): 
                        col = np.nan
                        row = np.nan
                
                if std_analysis:
                    if np.std(window_a)< std_threshold:
                        col = np.nan
                        row = np.nan
                
                
                u[k,m],v[k,m] = col, row
    
    return u, v*-1        
    

def remove_outliers(array, filter_size=5, sigma=1.5):
    returnarray = copy.copy(array)
    filter_diff = int(filter_size/2)
    #roll_arr = rolling_window(array, (window_size,window_size))
    bar = progressbar.ProgressBar(maxval=array.shape[0], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]) 
    bar.start()
    bar_iterator = 0
    for o in range(array.shape[0]):
        for p in range(array.shape[1]):
            act_px = array[o,p]
            try:
                act_arr = array[o-filter_diff:o+filter_diff+1,p-filter_diff:p+filter_diff+1]
                upperlim = np.nanmean(act_arr) + sigma*np.nanstd(act_arr)
                lowerlim = np.nanmean(act_arr) - sigma*np.nanstd(act_arr)
                
                if act_px< lowerlim or act_px> upperlim:
                    returnarray[o,p] = np.nanmean(act_arr)
            except:
                pass
        bar.update(bar_iterator+1)
        bar_iterator += 1
                
    bar.finish()
    return(returnarray)




def runTIVparallel(i, perturb, ws, ol, sa, olsa, method, rem_outliers = False, fiter_size=3, sigma=2, mean_analysis = False, std_analysis = False, std_threshold = 15):   

    u, v= window_correlation_tiv(frame_a=perturb[i], frame_b=perturb[i+1], window_size=ws, overlap_window=ol, overlap_search_area=olsa, 
                                 search_area_size_x=sa, corr_method=method, mean_analysis = mean_analysis, std_analysis = std_analysis, std_threshold = std_threshold)
         
    
    if rem_outliers:
        u = remove_outliers(u, filter_size=fiter_size, sigma=sigma)
        v = remove_outliers(v, filter_size=fiter_size, sigma=sigma)

    return(u,v)





