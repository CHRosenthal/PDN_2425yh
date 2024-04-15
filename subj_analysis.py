# -*- coding: utf-8 -*-
#!/usr/bin/env python3

'''
subj_analysis.py
This script estimates learning rate measures and extract 10-trial-block traces
'''

def main(args):
  import os
  import numpy as np
  import pandas as pd
  from fnmatch import fnmatch
  
  # fail-safe design
  args._in = os.path.realpath(args._in) + '/'
  args.out = os.path.realpath(args.out) + '/'
  args.idin = os.path.realpath(args.idin) + '/'
  if not os.path.isdir(args.out): os.mkdir(args.out)
  if not os.path.isdir(f'{args.out}block10'): os.mkdir(f'{args.out}block10')
  if not os.path.isdir(f'{args.out}onset'): os.mkdir(f'{args.out}onset')
  
  # progress check
  out_fname = f'{args.out}{args.subj}.txt'
  
  # types of trials to analyse
  type_list = ['baseline_us','baseline_cs1','baseline_2pair','baseline_2unp',
               'all_1order','us_1order','pr_1order',
               'rem_1order',
               'pr_2pair','all_2pair', # need to remove probe trials from 2_pair
               'all_2unp','pr_2unp',
               'ext_cs1','ext_2pair','ext_2unp']
  
  # params to estimate
  trial_count = []
  total_cr = []
  total_crcount = []
  mean_onset = []
  # mean_fwhm = []
  first_third_cr = []
  second_third_cr = []
  last_third_cr = []
  crcount_1 = [] # for onset calculations
  crcount_2 = []
  crcount_3 = []
  first_third_onset = []
  second_third_onset = []
  last_third_onset = []
  # first_third_fwhm = []
  # second_third_fwhm = []
  # last_third_fwhm = []
  first_third_height = []
  second_third_height = []
  last_third_height = []  
  first_10_cr = []
  max_10_cr = [] # highest CR% in 10 successive trials
  time_to_max_univ = [] # number of trials to the max 10-block, all types
  time_to_max_spec = [] # number of trials to the max 10-block of specific type
  last_10_cr = []
  time_to_th_univ = [] # number of trials to 40%
  time_to_th_spec = []
  
  # below vars initialised for the timing histogram
  timing_dflist = []
  
  # first trial of rem_1order
  try:
    rem_start = np.loadtxt(f'{args.idin}{args.subj}/{args.subj}_rem_1order.txt')[0]
  except: rem_start = 2147483647
  
  # identify pr_2pair trials
  try:
    pr2_id = np.loadtxt(f'{args.idin}{args.subj}/{args.subj}_pr_2pair.txt')
  except: pr2_id = []
  
  # identify range of 2_pair learning phase
  try:
    pair_id = np.loadtxt(f'{args.idin}{args.subj}/{args.subj}_all_2pair.txt')
  except: pair_id = [0]
  
  try:
    unp_id = np.loadtxt(f'{args.idin}{args.subj}/{args.subj}_all_2unp.txt')
    unp_id = unp_id[unp_id > pair_id[0]] # after the first batch of paired trials
    pair_thr = [pair_id[0],unp_id[-1]] # the first batch of paired trials ~ learning
  except: pair_thr = [pair_id[0],2147483647]
  
  # for one subject, the CS2P-CS1 training was erroneously 
  # interrupted by CS1-US training before the extinction session
  # this deviated from the experimental protocol
  # but for fair comparison, I still take the trials before the interruption
  # block for analysis.
  if args.subj == 'oed': pair_thr = [pair_id[0],631]
  
  for t in type_list:
    try:
      # read files
      trial_score = pd.read_table(f'{args._in}{args.subj}/{args.subj}_{t}_score.txt')
      if not fnmatch(t,'*baseline*'):
        trial_id = np.loadtxt(f'{args.idin}{args.subj}/{args.subj}_{t}.txt')
      else: trial_id = np.arange(trial_score.shape[0])
    except:
      # file not found
      trial_count.append(np.nan)
      total_cr.append(np.nan)
      total_crcount.append(np.nan)
      mean_onset.append(np.nan)
      # mean_fwhm.append(np.nan)
      first_third_cr.append(np.nan)
      second_third_cr.append(np.nan)
      last_third_cr.append(np.nan)
      crcount_1.append(np.nan)
      crcount_2.append(np.nan)
      crcount_3.append(np.nan)
      first_third_onset.append(np.nan)
      second_third_onset.append(np.nan)
      last_third_onset.append(np.nan)
      first_third_height.append(np.nan)
      second_third_height.append(np.nan)
      last_third_height.append(np.nan)
      # first_third_fwhm.append(np.nan)
      # second_third_fwhm.append(np.nan)
      # last_third_fwhm.append(np.nan)
      first_10_cr.append(np.nan)
      max_10_cr.append(np.nan)
      time_to_max_univ.append(np.nan)
      time_to_max_spec.append(np.nan)
      last_10_cr.append(np.nan)
      time_to_th_univ.append(np.nan)
      time_to_th_spec.append(np.nan)
      print()
      print(f'WARNING: for {args.subj}, no trials found for {t}')
      print()
      continue
      
    if fnmatch(t,'*1order*') and not fnmatch(t,'*rem*'):
      trial_filter = trial_id < rem_start
    elif fnmatch(t,'*2pair*') and not fnmatch(t,'*ext*') and not fnmatch(t,'*baseline*'):
      trial_filter = (trial_id >= pair_thr[0]) * (trial_id < pair_thr[1])
      if fnmatch(t,'*all_2pair*'): # remove probe trials
        trial_filter *= ~np.isin(trial_id, pr2_id)
      trial_filter = trial_filter.astype('?')
    else: 
      trial_filter = np.ones(trial_id.size).astype('?')
      
    trial_id = trial_id[trial_filter]
    trial_score = trial_score.loc[trial_filter,:]
    trial_score['trial_type'] = t
    
    # count number of trials
    n = trial_score.shape[0]
    trial_count.append(n)
    
    # total cr%
    total_cr.append(trial_score['peak'].mean())
    total_crcount.append(trial_score['peak'].sum())
    
    # mean timing
    mean_onset.append(np.nanmean(trial_score['onset']))
    
    # first third
    first_third_cr.append(trial_score.loc[:int(n/3),'peak'].mean())
    crcount_1.append(trial_score.loc[:int(n/3),'peak'].sum())
    first_third_onset.append(np.nanmean(trial_score.loc[:int(n/3),'onset']))
    first_third_height.append(np.nanmean(trial_score.loc[:int(n/3),'height']))
    # first_third_fwhm.append(np.nanmean(trial_score.loc[:int(n/3),'fwhm']))
    timing_df = trial_score.loc[:int(n/3),['onset','trial_type']].dropna()
    timing_df['stage'] = 'early'
    timing_dflist.append(timing_df)
    
    # second third
    second_third_cr.append(trial_score.loc[int(n/3):int(2*n/3),'peak'].mean())
    crcount_2.append(trial_score.loc[int(n/3):int(2*n/3),'peak'].sum())
    second_third_onset.append(
      np.nanmean(trial_score.loc[int(n/3):int(2*n/3),'onset']))
    second_third_height.append(
      np.nanmean(trial_score.loc[int(n/3):int(2*n/3),'height']))
    # second_third_fwhm.append(
    #   np.nanmean(trial_score.loc[int(n/3):int(2*n/3),'fwhm']))
    timing_df = trial_score.loc[int(n/3):int(2*n/3),['onset','trial_type']].dropna()
    timing_df['stage'] = 'mid'
    timing_dflist.append(timing_df)
    
    # last third
    last_third_cr.append(trial_score.loc[int(2*n/3):,'peak'].mean())
    crcount_3.append(trial_score.loc[int(2*n/3):,'peak'].sum())
    last_third_onset.append(np.nanmean(trial_score.loc[int(2*n/3):,'onset']))
    last_third_height.append(np.nanmean(trial_score.loc[int(2*n/3):,'height']))
    # last_third_fwhm.append(np.nanmean(trial_score.loc[int(2*n/3):,'fwhm']))
    timing_df = trial_score.loc[int(2*n/3):,['onset','trial_type']].dropna()
    timing_df['stage'] = 'late'
    timing_dflist.append(timing_df)
    
    # processes 10-trial-blocks: first, last, max, time to max, time to threshold
    # also exports whole 10-trial-block traces for certain trialt types
    if n > 10:
      if fnmatch(t,'*1order*'): th = 4
      if fnmatch(t,'*2pair*') or fnmatch(t,'*2unp*'): th = 3
      sum_10_trials = np.zeros(n-9)
      peaks = trial_score['peak'].to_numpy()
      for i in range(10):
        sum_10_trials += peaks[i:n-9+i]
      first_10_cr.append(sum_10_trials[0])
      last_10_cr.append(sum_10_trials[-1])
      max_10_cr.append(sum_10_trials.max())
      ttm = sum_10_trials.argmax() + 9
      time_to_max_spec.append(ttm)
      time_to_max_univ.append(trial_id[ttm]-trial_id[0])
      if max(sum_10_trials) >= th:
        tt3 = np.argwhere(sum_10_trials >= th)[0][0] + 9
        time_to_th_spec.append(tt3)
        time_to_th_univ.append(trial_id[tt3]-trial_id[0])
      else:
        time_to_th_spec.append(np.nan)
        time_to_th_univ.append(np.nan)
        
      if t in ['all_1order','pr_1order','rem_1order',
               'pr_2pair','all_2pair','all_2unp','pr_2unp',
               'ext_cs1']:
        np.savetxt(f'{args.out}block10/{args.subj}_{t}.txt',
                   sum_10_trials[::10].astype(np.int8))
      
    else:
      first_10_cr.append(np.nan)
      last_10_cr.append(np.nan)
      max_10_cr.append(np.nan)
      time_to_max_spec.append(np.nan)
      time_to_max_univ.append(np.nan)
      time_to_th_spec.append(np.nan)
      time_to_th_univ.append(np.nan)
  
  out_df = pd.DataFrame(
    dict(trial_count = trial_count,
    total_cr = total_cr,
    crcount = total_crcount,
    mean_onset = mean_onset,
    # mean_fwhm = mean_fwhm,
    first_third_cr = first_third_cr,
    second_third_cr = second_third_cr,
    last_third_cr = last_third_cr,
    first_third_crcount = crcount_1,
    second_third_crcount = crcount_2,
    last_third_crcount = crcount_3,
    first_third_onset = first_third_onset,
    second_third_onset = second_third_onset,
    last_third_onset = last_third_onset,
    first_third_height = first_third_height,
    second_third_height = second_third_height,
    last_third_height = last_third_height,
    # first_third_fwhm = first_third_fwhm,
    # second_third_fwhm = second_third_fwhm,
    # last_third_fwhm = last_third_fwhm,
    first_10_cr = first_10_cr,
    max_10_cr = max_10_cr,
    time_to_max_univ = time_to_max_univ,
    time_to_max_spec = time_to_max_spec,
    last_10_cr = last_10_cr,
    time_to_th_univ = time_to_th_univ,
    time_to_th_spec = time_to_th_spec),
    index = type_list)
  
  out_df.to_csv(out_fname,header = True, index = True, sep = '\t')
  
  timing_df = pd.concat(timing_dflist)
  timing_df.to_csv(f'{args.out}onset/{args.subj}.txt', header = True,
                   index = False, sep = '\t')

if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser(description = 'this script estimates the learning params')
  parser.add_argument('subj', help = 'subject ID')
  parser.add_argument('-i','--in', dest = '_in', default = '../score',
                      help = 'input directory, contains all subjects')
  parser.add_argument('-o','--out',dest = 'out', default = '../pheno',
                      help = 'output directory for trial data')
  parser.add_argument('--idin', dest = 'idin', default = '../trial_ids',
                      help = 'output directory for trial id/order')
  # always overwrites
  args = parser.parse_args()
  main(args)