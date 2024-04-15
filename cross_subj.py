# -*- coding: utf-8 -*-
#!/usr/bin/env python3
'''
cross_subj.py
This script conducts cross-subject analysis
'''

def main(args):
  import os
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  import seaborn as sns
  from fnmatch import fnmatch
  from time import perf_counter as t
  
  # master timer
  tic = t()
  
  # fail-safe design
  args._in = os.path.realpath(args._in) + '/'
  args.out = os.path.realpath(args.out) + '/'
  if not os.path.isdir(args.out): os.system(f'mkdir -p {args.out}')
  
  # subjects list
  subjs = pd.read_table(args.subj,index_col = None).sort_values(by = 'subj_id')
  
  # trial types
  type_list = ['baseline_us','baseline_cs1','baseline_2pair','baseline_2unp',
               'all_1order','us_1order','pr_1order',
               'rem_1order',
               'pr_2pair','all_2pair', # NB actually compounds
               'all_2unp','pr_2unp',
               'ext_cs1','ext_2pair','ext_2unp']
  
  ## SECTION 1: generate summary files
  out = f'{args.out}/summary'
  if not os.path.isdir(out): os.mkdir(out)
  
  # for participants that completed the study in two sessions, 
  # only the first exposure to each trial type is counted
  dflist = []
  for name,subj in zip(subjs['subj_ab'],subjs['subj_id']):
    df = pd.read_table(f'{args._in}{name}.txt', index_col = 0)
    if os.path.isfile(f'{args._in}{name}_old.txt'):
      df_old = pd.read_table(f'{args._in}{name}_old.txt', index_col = 0)
      for i in range(df.shape[0]):
        if not all(df_old.iloc[i,:].isna()): # found in the first session
          df.iloc[i,:] = df_old.iloc[i,:]
    df['trial_type'] = df.index
    df = df.melt(id_vars = 'trial_type')
    df['subj'] = subj
    dflist.append(df)
  master_df = pd.concat(dflist, axis = 0)
  
  # baseline
  baseline_filter = master_df.trial_type.isin(
    ['baseline_us','baseline_cs1','baseline_2pair','baseline_2unp'])
  baseline_filter *= master_df.variable.isin(['total_cr'])
  baseline_df = master_df.loc[baseline_filter,['subj','trial_type','value']].pivot_table(
    values = 'value', index = 'subj', columns = 'trial_type')
  baseline_df *= 10
  baseline_df = baseline_df.astype(np.int8)
  baseline_df = baseline_df.rename_axis(None)
  baseline_df.columns = ['baseline_us','baseline_cs1','baseline_2pair','baseline_2unp']
  baseline_df.to_csv(f'{out}/baseline.txt', sep = '\t', index = True, header = True)
  toc = t() - tic
  print(f'Generated baseline. time = {toc:.3f}')
  
  # extract mean onset and total CR%
  cr_df = master_df.loc[master_df.variable.isin(['total_cr']),:]
  cr_df = cr_df.pivot_table(values = 'value', index = 'subj', columns = 'trial_type')
  cr_df = cr_df.rename_axis(None)
  cr_df.columns = sorted(type_list)
  cr_df.to_csv(f'{out}/total_cr.txt', sep = '\t', index = True, header = True)
  
  onset_df = master_df.loc[master_df.variable.isin(['mean_onset']),:]
  onset_df = onset_df.pivot_table(values = 'value', index = 'subj', columns = 'trial_type')
  onset_df = onset_df.rename_axis(None)
  onset_df.columns = sorted(type_list)
  onset_df.to_csv(f'{out}/mean_onset.txt', sep = '\t', index = True, header = True)
  toc = t() - tic
  print(f'Processed mean CR% and timing. time = {toc:.3f}')
  
  # for other trial types, export: CR% by third, timing by third, and time to threshold
  cr_filter = master_df.variable.isin(['trial_count','total_cr','first_third_cr',
                                       'second_third_cr','last_third_cr'])
  onset_filter = master_df.variable.isin(
    ['crcount','mean_onset','first_third_onset','first_third_crcount',
     'second_third_onset','second_third_crcount','last_third_onset','last_third_crcount'])
  # fwhm_filter = master_df.variable.isin(['trial_count','mean_onset','first_third_fwhm',
  #                                      'second_third_fwhm','last_third_fwhm'])
  ttt_filter = master_df.variable.isin(
    ['time_to_max_univ','time_to_max_spec','time_to_th_univ','time_to_th_spec'])
  for x in ['all_1order','us_1order','pr_1order',
            'rem_1order',
            'pr_2pair','all_2pair', # NB actually compounds
            'all_2unp','pr_2unp',
            'ext_cs1','ext_2pair','ext_2unp']:
    x_filter = master_df.trial_type.isin([x])
    cr_df = master_df.loc[x_filter*cr_filter,['subj','variable','value']].pivot_table(
      values = 'value', index = 'subj', columns = 'variable').rename_axis(None)
    cr_df = cr_df[['trial_count','total_cr','first_third_cr','second_third_cr','last_third_cr']]
    cr_df.to_csv(f'{out}/{x}_cr.txt', sep = '\t', index = True, header = True)
    
    onset_df = master_df.loc[x_filter*onset_filter,['subj','variable','value']].pivot_table(
      values = 'value', index = 'subj', columns = 'variable').rename_axis(None)
    onset_df = onset_df[['crcount','mean_onset','first_third_onset','first_third_crcount',
     'second_third_onset','second_third_crcount','last_third_onset','last_third_crcount']]
    onset_df.to_csv(f'{out}/{x}_onset.txt', sep = '\t', index = True, header = True)
    
    # fwhm_df = master_df.loc[x_filter*fwhm_filter,['subj','variable','value']].pivot_table(
    #   values = 'value', index = 'subj', columns = 'variable').rename_axis(None)
    # fwhm_df = fwhm_df[['trial_count','mean_fwhm','first_third_fwhm','second_third_fwhm','last_third_fwhm']]
    # fwhm_df.to_csv(f'{out}/{x}_fwhm.txt', sep = '\t', index = True, header = True)
    
    ttt_df = master_df.loc[x_filter*ttt_filter,['subj','variable','value']].pivot_table(
      values = 'value', index = 'subj', columns = 'variable').rename_axis(None)
    ttt_df = ttt_df[['time_to_max_univ','time_to_max_spec','time_to_th_univ','time_to_th_spec']]
    ttt_df.to_csv(f'{out}/{x}_ttt.txt', sep = '\t', index = True, header = True)
    
    
    toc = t() - tic
    print(f'Processed summary of {x}. time = {toc:.3f}')
   
  ## SECTION 2: 10_trial_blocks
  out = f'{args.out}/block10'
  if not os.path.isdir(out): os.mkdir(out)
  os.chdir(args._in); os.chdir('block10')
  
  for x in ['all_1order','pr_1order','rem_1order',
           'pr_2pair','all_2pair','all_2unp',
           'ext_cs1']:
    flist = []
    for name in subjs['subj_ab']:
      old_session = False
      for f in os.listdir():
        if fnmatch(f,f'*{x}*') and fnmatch(f,f'*{name}*') and fnmatch(f,'*old*'):
          flist.append(f) # prioritise first session
          old_session = True
      if not old_session: # then search for latest session
        for f in os.listdir():
          if fnmatch(f,f'*{x}*') and fnmatch(f,f'*{name}*'):
            flist.append(f)
    
    fig, ax = plt.subplots(figsize = (7,4))
    for f,_id in zip(flist,subjs['subj_id']):
      trace = np.loadtxt(f,dtype = np.float32)
      if fnmatch(f,'*1order*'):
        baseline = baseline_df.loc[_id,'baseline_cs1']
      elif fnmatch(f,'*2pair*'):
        baseline = baseline_df.loc[_id,'baseline_2pair']
      elif fnmatch(f,'*2unp*'):
        baseline = baseline_df.loc[_id,'baseline_2unp']
      
      if fnmatch(f,'*ext*') or fnmatch(f,'*all_2pair*') or fnmatch(f,'*rem*'):
        ax.plot(np.arange(1,len(trace)+1),trace - 0.2 + 0.05*_id, # slight offset
                marker = '.', label = f'subj {_id}')
      else:
        trace = np.concatenate([[baseline],trace])
        ax.plot(np.concatenate(([-3],np.arange(1,len(trace)))), 
                trace - 0.2 + 0.05*_id, marker = '.', label = f'subj {_id}')
    
    if not fnmatch(f, '*ext*'):
      ax.axvline(x = 0, color = 'k')
      ticks = np.array(ax.get_xticks())
      ticks = ticks[ticks > 0]
      ticks = np.concatenate(([-3],ticks))
      ticklbl = ticks.astype(np.int32).astype('U'); ticklbl[0] = 'baseline'
      ax.set_xticks(ticks, ticklbl)
    
    ax.set_xlabel('10-trial blocks')
    ax.set_ylabel('Number of CR in a 10-trial block')
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(f'{out}/{x}.pdf', bbox_inches = 'tight')
    plt.savefig(f'{out}/{x}.png')
    plt.close()
    
    toc = t() - tic
    print(f'Processed 10-block traces for {x}. time = {toc:.3f}')
  
  ## SECTION 3: RESPONSE LATENCY HISTOGRAMS
  out = f'{args.out}/onset_hist'
  if not os.path.isdir(out): os.mkdir(out)
  if not os.path.isdir(f'{out}/subj_wise'): os.mkdir(f'{out}/subj_wise')
  if not os.path.isdir(f'{out}/type_by_subj'): os.mkdir(f'{out}/type_by_subj')
  if not os.path.isdir(f'{out}/type_by_stage'): os.mkdir(f'{out}/type_by_stage')
  os.chdir(args._in); os.chdir('onset')
  
  dflist = []
  for name,subj in zip(subjs['subj_ab'],subjs['subj_id']):
    df = pd.read_table(f'{name}.txt', index_col = None)
    if os.path.isfile(f'{name}_old.txt'):
      df_old = pd.read_table(f'{name}_old.txt', index_col = None)
      # exclude trials in 2nd exposure
      df = df.loc[~df['trial_type'].isin(df_old['trial_type'].unique()),:]
      df = pd.concat((df_old,df),axis = 0)
    bl_filter = []
    for x in df['trial_type'].unique():
      if fnmatch(x,'*baseline*'): bl_filter.append(x)
    df = df.loc[~df['trial_type'].isin(bl_filter),:]
    df['subj'] = subj
    dflist.append(df)
  master_df = pd.concat(dflist, axis = 0)
  master_df.to_csv(f'{out}/all_cr_onset.txt', sep = '\t', index = False, header = True)
  
  # subject-wise plotting
  for name,subj in zip(subjs['subj_ab'],subjs['subj_id']):
    subj_df = master_df.loc[master_df['subj']==subj,:]
    for x in subj_df['trial_type'].unique():      
      onset_df = subj_df.loc[subj_df.trial_type == x,:]
      _,ax = plt.subplots(figsize = (4,4))
      sns.histplot(onset_df, x = 'onset', hue = 'stage', multiple = 'stack', ax = ax)
      ax.set_title(f'Subj {subj:.0f}')
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
      plt.savefig(f'{out}/subj_wise/{name}_{x}.png')
      plt.savefig(f'{out}/subj_wise/{name}_{x}.pdf',bbox_inches = 'tight')
      plt.close()
    
    toc = t() - tic
    print(f'Processed timing histograms for {name}. time = {toc:.3f}')
      
  # trial-type-specific plotting
  for x in master_df['trial_type'].unique():
    if fnmatch(x,'*1order*'): c = '#ffc000'
    if fnmatch(x,'*2pair*'): c = '#c00000'
    if fnmatch(x,'*2unp*'): c = '#0070c0'
    palette = sns.light_palette(c,as_cmap = True)
    
    type_df = master_df.loc[master_df['trial_type']==x,:]
    _,ax = plt.subplots(figsize = (4,4))
    sns.histplot(type_df, x = 'onset', hue = 'subj', multiple = 'stack', ax = ax,
                 palette = palette)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_legend().remove()
    plt.savefig(f'{out}/{x}.png')
    plt.savefig(f'{out}/{x}.pdf',bbox_inches = 'tight')
    plt.close()
    
    _,ax = plt.subplots(figsize = (4,4))
    sns.histplot(type_df, x = 'onset', hue = 'stage', multiple = 'stack', ax = ax)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(f'{out}/type_by_stage/{x}.png')
    plt.savefig(f'{out}/type_by_stage/{x}.pdf',bbox_inches = 'tight')
    plt.close()
    
    stages = ['early','mid','late']
    _, ax = plt.subplots(1,len(stages),figsize = (4*len(stages),4))
    for i in range(len(stages)):
      onset_df = type_df.loc[type_df.stage==stages[i],:]
      try:
        sns.histplot(onset_df, x = 'onset', hue = 'subj', multiple = 'stack', 
                     ax = ax[i], palette = palette)
      except: ax[i].text(0,0,'NO DATA')
      ax[i].spines['right'].set_visible(False)
      ax[i].spines['top'].set_visible(False)
      ax[i].get_legend().remove()
      ax[i].set_title(f'{stages[i]} acquisition')
    plt.savefig(f'{out}/type_by_subj/stages_{x}.png')
    plt.savefig(f'{out}/type_by_subj/stages_{x}.pdf',bbox_inches = 'tight')
    plt.close()
    
    toc = t() - tic
    print(f'Processed timing histograms for {x}. time = {toc:.3f}')
  
if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser(description = 'Cross-subject analysis - batch script')
  parser.add_argument('-s','--subj', dest = 'subj', default = '../params/subjs.txt',
                      help = 'list of subjs and grouping')
  parser.add_argument('-i','--in', dest = '_in', default = '../pheno',
                      help = 'input directory')
  parser.add_argument('-o','--out',dest = 'out', default = '../stats',
                      help = 'output directory')
  parser.add_argument('-f','--force', action = 'store_true', default = False, 
                      help = 'force overwrite')
  args = parser.parse_args()
  main(args)