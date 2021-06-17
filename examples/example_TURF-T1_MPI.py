#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:35:42 2021

@author: benjamin
"""

### MPI Example for HPCs with MPI structure ###

import os
os.chdir("..")
print(os.getcwd())
import sys
sys.path.insert(1, os.getcwd())

### IMPORTANT: Please make sure that the current working directory is set to this files' path!
from functions.ATIV_fun import *
#from joblib import Parallel, delayed
from mpi4py import MPI
import numpy as np
import time




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
time_lst = [60]



wd_path = os.getcwd()
data_path = os.path.join(wd_path, "data")
example_file = os.path.join(data_path, "TURF-T1_Tb_stab.npy")

outpath = data_path+"/"

tb = np.load(example_file)
ret_lst = randomize_find_interval(data = tb,rec_freq = 2)
time_interval = ret_lst[0]







#uas_lst = []
#vas_lst = []
start_time = time.time()
#for t in time_lst:
t = 60    
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



perturb = create_tst_perturbations_spmm(tb,t)
    
if set_len is not None:
	len_perturb = set_len
else:
	len_perturb = len(perturb)



#seeds = 5000

#If the rank is 0 (master) then create a list of numbers from 0-4999
#and then split those seeds number equally amoung size groups,
#other set seeds and split_seeds to $
if rank == 0:
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
    seeds = np.arange(len_perturb)
    split_seeds = np.array_split(seeds, size, axis = 0)
    	

else:
	seeds = None
	split_seeds = None

rank_seeds = comm.scatter(split_seeds, root = 0)

rank_data = []

for i in np.arange(len(rank_seeds)):    
        seed = rank_seeds[i]
        np.random.seed(seed)
        
        u, v = runTIVparallel(i, perturb=perturb, ws=ws, ol=ol, sa=sa, olsa=olsa, method=method)
        rank_data.append([u, v])

data_gather = comm.gather(rank_data, root = 0)

if rank == 0:
    
    out_uv = np.array(data_gather)        
    
    
    uas = out_uv[:,0,:,:]
    vas = out_uv[:,1,:,:]
    np.save(outpath+"UAS_"+experiment+"_"+str(t)+".npy", uas)
    np.save(outpath+"VAS_"+experiment+"_"+str(t)+".npy", vas)        
print("Elapsed time (sec):")
print(time.time() - start_time)
           
