# -*- coding: utf-8 -*-
#!/usr/bin/env python3

'''
denoise.py
This script de-noises data
'''

# inputs are all 2000-element time traces
# element 1500 is the US
# first 500 ms are the baseline

def main(args):
  import os
  if os.path.isfile(args.out) and not args.force: return
  import numpy as np
  import scipy.stats as sts
  import scipy.signal as sgn
  from time import perf_counter as t
  
  def denoise(trial):
    # normalise data
    tmp = np.round(trial,2)[900:1600]
    trial -= sts.mode(tmp).mode
    del tmp
    
    # low pass filter - let's say that no eyeblinks take less than 20 ms (50 Hz)
    trial_lowpass = sgn.sosfilt(sgn.butter(4,50,fs=1000,output='sos'),trial)
    
    # spline filter
    filtered = sgn.cspline1d_eval(sgn.cspline1d(trial_lowpass), 
                                  np.arange(len(trial_lowpass)))
    return filtered
  
  tic = t()
  
  x = np.loadtxt(args._in, delimiter = '\t')
  x_copy = x.copy()
  filtered_x = np.zeros(x.shape)
  for i in range(x.shape[0]):
    if i % 100 == 0:
      toc = t() - tic
      print(f'{i}/{x.shape[0]} trials processed, time = {toc:.3f}')
    
    trial = x[i,:]
    filtered_x[i,:] = denoise(trial)
  
  np.savetxt(args.out, filtered_x, fmt = '%.7f', delimiter = '\t')
  
  # output diagnostic figs
  import matplotlib.pyplot as plt
  _,ax = plt.subplots(2,1,figsize = (12,6))
  t = np.arange(-1550,450)
  orig_mean = x_copy.mean(axis = 0); orig_std = x_copy.std(axis = 0)
  new_mean = filtered_x.mean(axis = 0); 
  new_max = filtered_x.max(axis = 0)
  new_min = filtered_x.min(axis = 0)
  ax[0].plot(t,orig_mean,'b')
  ax[0].fill_between(t, orig_mean-orig_std, orig_mean+orig_std, color='b', alpha = .2)
  ax[0].set_title('original')
  ax[1].plot(t,new_mean,'r')
  ax[1].fill_between(t, new_min, new_max, color = 'r', alpha = .2)
  ax[1].set_title('denoised')
  plt.savefig(args.out.replace('.txt','_diag.png'))
  plt.close()
  
  _,ax = plt.subplots(figsize=(6,4))
  trial_id = np.random.randint(0,x_copy.shape[0])
  orig = x_copy[trial_id,:]; filtered = filtered_x[trial_id,:]
  orig -= orig.mean(); filtered -= filtered.mean()
  ax.plot(t, orig,'k',label = 'original')
  ax.plot(t,filtered,'r',label = 'filtered')
  ax.axvspan(-1300,-1250, fc = '#ffffdf', color = '#ffffdf')
  inset = ax.inset_axes([0.2,0.7,0.2,0.15], title = 'zoom-in for -1300 to -1250 ms')
  inset.plot(t[250:300],orig[250:300],'k')
  inset.plot(t[250:300],filtered[250:300],'r')
  plt.savefig(args.out.replace('.txt','.png'))
  plt.close()
    
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description = 'de-noises a single file')
  parser.add_argument('-i','--in', dest = '_in', 
                      help = 'input txt file, each row = 1 trial, each column = 1 ms')
  parser.add_argument('-o','--out',dest = 'out',
                      help = 'output file')
  parser.add_argument('-f','--force', action = 'store_true', default = False, help = 'force overwrite')
  args = parser.parse_args()
  main(args)