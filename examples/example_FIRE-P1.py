#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 14:44:07 2021

@author: benjamin
"""

import os

### IMPORTANT: Please make sure that the current working directory is set to this files' path!
from functions.ATIV_fun import *
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt 


experiment = "FIRE-2019P1"
experiment_num = 1
ws=16
ol = 15
sa = 24
olsa = 22
method = "greyscale"

rec_freq = 3

set_len = 4
# times according to Inagaki 2013, here *2 due to the recording frequency
time_lst = None



wd_path = os.getcwd()
data_path = os.path.join(wd_path, "data")
example_file = os.path.join(data_path, "FIRE_P1_Tb_stab.npy")

outpath = data_path+"/"

tb = np.load(example_file)
time_interval = 1

f= open(outpath+"FIRE_P1-TIV_metadata.txt","w+")
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




if set_len is not None:
    len_tb = set_len
out_lst = Parallel(n_jobs=-1)(delayed(runTIVparallel)(i, perturb=tb, ws=ws, ol=ol, sa=sa, olsa=olsa, method=method, rem_outliers = True, fiter_size=3, sigma=2, mean_analysis = True, std_analysis = True, std_threshold = 15) for i in range(0, len_tb))
out_uv = np.array(out_lst)
uas = out_uv[:,0,:,:]
vas = out_uv[:,1,:,:]
np.save(outpath+"UAS_"+experiment+"_.npy", uas)
np.save(outpath+"VAS_"+experiment+"_.npy", vas)



x, y = get_coordinates(image_size=tb[0].shape, window_size=sa, overlap=olsa )    
plt.imshow(tb[2], cmap="RdBu_r")
plt.quiver(x,y,np.flipud(np.round(uas[2],2)),np.flipud(np.round(vas[2],2)))
plt.close()