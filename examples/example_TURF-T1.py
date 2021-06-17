#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:35:42 2021

@author: benjamin
"""



import os

### IMPORTANT: Please make sure that the current working directory is set to this files' path!
from functions.ATIV_fun import *
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt 

experiment = "TURF-T1"
experiment_num = 1
ws=32
ol = 28
sa = 64
olsa = 60
method = "greyscale"

rec_freq = 2

set_len = 4
# times according to Inagaki 2013, here *2 due to the recording frequency
time_lst = [60, 40, 20, 10]



wd_path = os.getcwd()
data_path = os.path.join(wd_path, "data")
example_file = os.path.join(data_path, "TURF-T1_Tb_stab.npy")

outpath = data_path+"/"

tb = np.load(example_file)
ret_lst = randomize_find_interval(data = tb,rec_freq = rec_freq)
time_interval = ret_lst[0]

f= open(outpath+"ATIV_metadata.txt","w+")
f.write("METADATA "+experiment+" E"+str(experiment_num)+" A-TIV \r\n") 
f.write("SETTINGS: \r\n") 
f.write("ws="+str(ws)+" \r\n")
f.write("ol="+str(ol)+" \r\n")
f.write("sa="+str(sa)+" \r\n")
f.write("olsa="+str(olsa)+" \r\n")
f.write("method="+str(method)+" \r\n")
f.write("time_lst="+str(time_lst)+" \r\n")
f.write("time_interval="+str(time_interval)+" \r\n")
f.write("set_len="+str(set_len)+" \r\n")    
f.close()





uas_lst = []
vas_lst = []
for time in time_lst:
    
    print("Creating Perturbations.. \n")
    
    perturb = create_tst_perturbations_mm(tb,time) 
    #pertub = create_tst_perturbations_spmm(tb,time)
    
    
    len_perturb = len(perturb)
    
    if set_len is not None:
        len_perturb = set_len
    out_lst = Parallel(n_jobs=-1)(delayed(runTIVparallel)(i, interval=time_interval, perturb=perturb, ws=ws, ol=ol, sa=sa, olsa=olsa, method=method) for i in range(0, len_perturb))
    out_uv = np.array(out_lst)
    uas = out_uv[:,0,:,:]
    vas = out_uv[:,1,:,:]
    np.save(outpath+"UAS_"+experiment+"_"+str(time)+".npy", uas)
    np.save(outpath+"VAS_"+experiment+"_"+str(time)+".npy", vas)
    uas_lst.append(uas)
    vas_lst.append(vas)


# Note: The filling may be done in a seperate file/process

uas_full = fill_weight(arr_lst=uas_lst, time_lst=time_lst)
vas_full = fill_weight(arr_lst=vas_lst, time_lst=time_lst)

# Example plot:

perturb_0 = create_tst_perturbations_mm(tb,moving_mean_size=time_lst[0])

# Artificial pattern created by people with moving metal plates 

x, y = get_coordinates(image_size=perturb_0[0].shape, window_size=sa, overlap=olsa )    
plt.imshow(perturb_0[2], vmin = -1, vmax=1, cmap="RdBu_r")
plt.quiver(x,y,np.flipud(np.round(uas_full[2],2)),np.flipud(np.round(vas_full[2],2)))
plt.close()