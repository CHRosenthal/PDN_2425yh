# -*- coding: utf-8 -*-
#!/usr/bin/env python3

'''
baseline_extract.py
This script pre-processes spike2 data and extracts baseline trials
'''

def main(args):
  import os
  import numpy as np
  import pandas as pd
  from fnmatch import fnmatch
  
  
  # fail-safe design
  args._in = os.path.realpath(args._in) + '/'
  args.out = os.path.realpath(args.out) + '/'
  
  # trial types
  trial_types = ['baseline_us','baseline_cs1','baseline_2pair','baseline_2unp']
  
  # progress check
  skip = True
  for x in trial_types:
    if not os.path.isfile(f'{args.out}{args.subj}/{x}.txt'): skip = False
  if skip and not args.force: return
  
  # makes output directory
  os.chdir(args._in)
  if not os.path.isdir(args.out): os.mkdir(args.out)
  
  # input raw spike2 exports
  df = pd.read_csv(f'{args.subj}_baseline.txt', sep = '\t')
  
  # output initialisation
  bl_us = []
  bl_cs1 = []
  bl_2pair = []
  bl_2unp = []
  
  # rename columns
  col = df.columns.tolist()
  for i in range(len(col)):
    if fnmatch(col[i],'*emg*'): col[i] = 'emg'
    if fnmatch(col[i],'*ark*'): col[i] = 'digmark'
  df.columns = col
  
  emg = df['emg'].to_numpy()
  tmp = np.repeat(emg[0],500) # some trials start within the first 1.5s and are skipped
  emg = np.concatenate((tmp,emg)) # the non-existent 500 ms will be cut out eventually
  trial_id = 0
  
  for i in range(1000,df.shape[0]-500):
    if df['digmark'][i] > 0:
      if trial_id < 10: bl_us.append(emg[i-1000:i+1000])
      elif trial_id < 20: bl_cs1.append(emg[i-1000:i+1000])
      elif trial_id < 30: bl_2pair.append(emg[i-1000:i+1000])
      else: bl_2unp.append(emg[i-1000:i+1000])
      trial_id += 1
      
  # output
  if not os.path.isdir(f'{args.out}{args.subj}'): os.mkdir(f'{args.out}{args.subj}')
  
  for var, name in zip([bl_us, bl_cs1,bl_2pair,bl_2unp],trial_types):
    stack = np.vstack(var)
    np.savetxt(f'{args.out}{args.subj}/{args.subj}_{name}.txt', stack, fmt ='%.7f', delimiter = '\t')
    
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description = 'pre-processes spike2 data and extracts trial traces')
  parser.add_argument('subj', help = 'subject ID')
  parser.add_argument('-i','--in', dest = '_in', default = '../baseline',
                      help = 'input directory, contains all subjects')
  parser.add_argument('-o','--out',dest = 'out', default = '../extract',
                      help = 'output directory for trial data')
  parser.add_argument('-f','--force', action = 'store_true', default = False, help = 'force overwrite')
  args = parser.parse_args()
  main(args)