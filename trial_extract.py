# -*- coding: utf-8 -*-
#!/usr/bin/env python3

'''
trial_extract.py
This script pre-processes spike2 data and extracts trial traces
'''

def main(args):
  import os
  import numpy as np
  import pandas as pd
  from fnmatch import fnmatch
  from time import perf_counter as t
  
  tic = t()
  
  # fail-safe design
  args._in = os.path.realpath(args._in) + '/'
  args.out = os.path.realpath(args.out) + '/'
  args.idout = os.path.realpath(args.idout) + '/'
  
  # trial types
  trial_types = ['us_1order','pr_1order','rem_1order','all_1order','im_2pair',
   'pr_2pair','all_2pair','im1_2unp','im2_2unp','pr_2unp','all_2unp',
   'ext_cs1','ext_2unp','ext_2pair']
  
  # progress check
  skip = True
  for x in trial_types:
    if not os.path.isfile(f'{args.out}{args.subj}/{x}.txt'): skip = False
    if not os.path.isfile(f'{args.idout}{args.subj}/{x}.txt'): skip = False
  if skip and not args.force: return
  
  # makes output directory
  os.chdir(args._in)
  if not os.path.isdir(args.out): os.mkdir(args.out)
  if not os.path.isdir(args.idout): os.mkdir(args.idout)
  
  # input raw spike2 exports
  os.chdir(args.subj)
  flist = []
  for x in os.listdir():
    if fnmatch(x, '*.txt'):
      flist.append(x)
  
  # ensure all files are in the correct order
  flist.sort()
  
  # output initialisation
  us_1order = []; us_1order_id = []
  pr_1order = []; pr_1order_id = []
  rem_1order = []; rem_1order_id = []
  all_1order = []; all_1order_id = []
  im_2pair = []; im_2pair_id = []
  pr_2pair = []; pr_2pair_id = []
  all_2pair = []; all_2pair_id = []
  im1_2unp = []; im1_2unp_id = []
  im2_2unp = []; im2_2unp_id = []
  pr_2unp = []; pr_2unp_id = []
  all_2unp = []; all_2unp_id = []
  ext_cs1 = []; ext_cs1_id = []
  ext_2unp = []; ext_2unp_id = []
  ext_2pair = []; ext_2pair_id = []
  
  count = 0
  trial_id = 0 # count the total number of trials, never gets reset
  for x in flist:
    count += 1
    df = pd.read_csv(x, sep = '\t')
    # rename columns
    col = df.columns.tolist()
    for i in range(len(col)):
      if fnmatch(col[i],'*Time*'): col[i] = 'time'
      if fnmatch(col[i],'*1order_us'): col[i] = 'us1'
      if fnmatch(col[i],'*1order_im'): col[i] = 'pr1'
      if fnmatch(col[i],'*2rem_us'): col[i] = 'rem1'
      if fnmatch(col[i],'*2pair_im'): col[i] = 'im2'
      if fnmatch(col[i],'*2pair_pr'): col[i] = 'pr2'
      if fnmatch(col[i],'*2unp_im2'): col[i] = 'im3'
      if fnmatch(col[i],'*2unp_im1'): col[i] = 'im1'
      if fnmatch(col[i],'*2unp_pr'): col[i] = 'pr3'
      if fnmatch(col[i],'*emg*'): col[i] = 'emg'
      if fnmatch(col[i],'*1ext_im'): col[i] = 'ex1'
      if fnmatch(col[i],'*2pair_ext'): col[i] = 'ex2'
      if fnmatch(col[i],'*2unp_ext'): col[i] = 'ex3'
    df.columns = col
    
    # supply zeroes to empty columns
    for i in ['time','us1','pr1','rem1','im2','im3','im1','pr3','emg','ex1','ex2','ex3']:
      if not i in df.columns:
        df[i] = 0
    
    # regularise time to ms
    df['time'] = (1000 * df.time).astype(np.uint32)
    
    # quality control
    if df['time'][1] - df['time'][0] != 1:
      raise ValueError(f'Wrong sampling rate for {x}, please export from spike2 again.')
    
    # for easy handling
    emg = df['emg']
    # classify data
    for i in range(1500,df.shape[0]-501):
      if df['us1'][i] and not df['pr1'][i]:
        trial_id += 1;
        us_1order.append(emg[i-1500:i+500]); all_1order.append(emg[i-1500:i+500]);
        us_1order_id.append(trial_id); all_1order_id.append(trial_id)
      if df['pr1'][i]:
        trial_id += 1;
        pr_1order.append(emg[i-1500:i+500]);all_1order.append(emg[i-1500:i+500])
        pr_1order_id.append(trial_id); all_1order_id.append(trial_id)
      if df['rem1'][i]:
        trial_id += 1
        rem_1order.append(emg[i-1500:i+500]); all_1order.append(emg[i-1500:i+500])
        rem_1order_id.append(trial_id); all_1order_id.append(trial_id)
      if df['im2'][i] and not df['pr2'][i]: 
        trial_id += 1
        im_2pair.append(emg[i-1500:i+500]); all_2pair.append(emg[i-1500:i+500])
        im_2pair_id.append(trial_id); all_2pair_id.append(trial_id)
      if df['pr2'][i]: 
        trial_id += 1
        pr_2pair.append(emg[i-1500:i+500]); all_2pair.append(emg[i-1500:i+500])
        pr_2pair_id.append(trial_id); all_2pair_id.append(trial_id)
      if df['im1'][i]: 
        trial_id += 1
        im1_2unp.append(emg[i-1500:i+500])
        im1_2unp_id.append(trial_id)
      if df['im3'][i] and not df['pr3'][i]: 
        trial_id += 1
        im2_2unp.append(emg[i-1500:i+500]); all_2unp.append(emg[i-1500:i+500])
        im2_2unp_id.append(trial_id); all_2unp_id.append(trial_id)
      if df['pr3'][i]: 
        trial_id += 1
        pr_2unp.append(emg[i-1500:i+500]); all_2unp.append(emg[i-1500:i+500])
        pr_2unp_id.append(trial_id); all_2unp_id.append(trial_id)
      if df['ex1'][i]: 
        trial_id += 1
        ext_cs1.append(emg[i-1500:i+500]);
        ext_cs1_id.append(trial_id)
      if df['ex2'][i]: 
        trial_id += 1
        ext_2pair.append(emg[i-1500:i+500])
        ext_2pair_id.append(trial_id)
      if df['ex3'][i]: 
        trial_id += 1
        ext_2unp.append(emg[i-1500:i+500])
        ext_2unp_id.append(trial_id)
      
    # log
    toc = t() - tic
    print(f'Finished extractions from {x}, time = {toc:.3f} seconds, {count}/{len(flist)}')
    
  # output
  if not os.path.isdir(f'{args.out}{args.subj}'): os.mkdir(f'{args.out}{args.subj}')
  if not os.path.isdir(f'{args.idout}{args.subj}'): os.mkdir(f'{args.idout}{args.subj}')
  
  for var, varid, name in zip(
      [us_1order,pr_1order,rem_1order,all_1order,im_2pair,
       pr_2pair,all_2pair,im1_2unp,im2_2unp,pr_2unp,all_2unp,
       ext_cs1,ext_2unp,ext_2pair],
      [us_1order_id,pr_1order_id,rem_1order_id,all_1order_id,im_2pair_id,
       pr_2pair_id,all_2pair_id,im1_2unp_id,im2_2unp_id,pr_2unp_id,all_2unp_id,
       ext_cs1_id,ext_2unp_id,ext_2pair_id],
      trial_types):
    
    try:
      stack = np.vstack(var)
      varid = np.array(varid, dtype = np.int32)
      varid = varid.reshape((varid.size,1)) # force vertical
      np.savetxt(f'{args.out}{args.subj}/{args.subj}_{name}.txt', stack, fmt ='%.7f', delimiter = '\t')
      np.savetxt(f'{args.idout}{args.subj}/{args.subj}_{name}_id.txt',varid, fmt = '%i')
    except:
      if len(var) == 0:
        print(f'WARNING: for {args.subj}, there are no trials in category {name}')
      else: 
        print(var)
        raise
    
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description = 'pre-processes spike2 data and extracts trial traces')
  parser.add_argument('subj', help = 'subject ID')
  parser.add_argument('-i','--in', dest = '_in', default = '../raw',
                      help = 'input directory, contains all subjects')
  parser.add_argument('-o','--out',dest = 'out', default = '../extract',
                      help = 'output directory for trial data')
  parser.add_argument('--idout', dest = 'idout', default = '../trial_ids',
                      help = 'output directory for trial id/order')
  parser.add_argument('-f','--force', action = 'store_true', default = False, help = 'force overwrite')
  args = parser.parse_args()
  main(args)