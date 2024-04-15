# -*- coding: utf-8 -*-
#!/usr/bin/env python3
'''
stats_onset.py
This script performs a permutation test for timing across stages of the experiment
'''

def main(args):
  import pandas as pd
  import matplotlib.pyplot as plt
  import numpy as np
  import os
  from fnmatch import fnmatch
  from time import perf_counter as t
  
  rng = np.random.Generator(np.random.MT19937(310110))
  tic = t()
  
  # fail-safe design
  args._in = os.path.realpath(args._in)
  args.out = os.path.realpath(args.out) + '/'
  
  if not os.path.isdir(args.out): os.mkdir(args.out)
  
  # subjects list
  subjs = pd.read_table(args.subj,index_col = None).sort_values(by = 'subj_id')
  
  # subjects grouping
  psy_yes = subjs.loc[subjs.psy == 1, 'subj_id']
  psy_no = subjs.loc[subjs.psy == 0, 'subj_id']
  cs2p_first = subjs.loc[subjs.order == 'cs2p', 'subj_id']
  cs2u_first = subjs.loc[subjs.order == 'cs2u', 'subj_id']
  hand_l = subjs.loc[subjs.cs2p_hand == 'LEFT', 'subj_id']
  hand_r = subjs.loc[subjs.cs2p_hand == 'RIGHT', 'subj_id']
  male = subjs.loc[subjs.sex == 'M', 'subj_id'] 
  female = subjs.loc[subjs.sex == 'F', 'subj_id'] 
  
  # define the t statistic
  def tstat(a,b):
    a = np.array(a); b = np.array(b);
    s = a.mean()-b.mean()
    s = s / (a.var()*len(a) + b.var()*len(b))**.5 / (
      1/len(a) + 1/len(b))**.5 * (len(a)+len(b)-2)**.5
    return s
  
  # define the stratified bootstrap function
  def strat_bs(data1,data2, nbs = 65536):
    # must be pd.DataFrame, 1st column = data, 2nd column = group
    sqr_nbs = int(nbs**.5); nbs = sqr_nbs**2 # enforce integer n_bootstrap
    
    # g1 and g2 are groups of data, (i.e. subjects)
    # each bootstrap sample resamples from within each group
    d1 = data1.iloc[:,0].to_numpy(); g1 = data1.iloc[:,1].to_numpy()
    d2 = data2.iloc[:,0].to_numpy(); g2 = data2.iloc[:,1].to_numpy()
    obs_t = tstat(d1,d2)
    
    # transform data
    mu = (d1.sum()+d2.sum())/(len(d1)+len(d2))
    d1 += mu - d1.mean(); d2 += mu - d2.mean()
    
    bss1 = np.zeros((sqr_nbs,len(d1))); bss2 = np.zeros((sqr_nbs,len(d2)))
    for i in range(sqr_nbs):
      for j in np.unique(g1): # bootstrap for each stratified sample
        bss1[i,g1==j] = rng.choice(d1[g1==j],size=(g1==j).sum())
      for j in np.unique(g2):
        bss2[i,g2==j] = rng.choice(d2[g2==j],size=(g2==j).sum())
    
    bs_t = np.zeros((sqr_nbs,sqr_nbs))
    for i in range(sqr_nbs):
      for j in range(sqr_nbs):
        bs_t[i,j] = tstat(bss1[i,:],bss2[j,:])
    
    p = (bs_t < obs_t).sum()/nbs
    p = min((p,1-p)) * 2 # two-tailed
    return p,obs_t
  
  # convert CI level to significance labels
  def siglabel(p):
    if p > 0.1: return 'ns'
    elif p > 0.05: return f'p = {p:.3f}'
    elif p > 0.01: return '*'
    elif p > 10**-3: return '**'
    elif p > 10**-4: return '***'
    else: return '****'
  
  # summary stats log
  master_log = open(f'{args.out}descriptive_stats.log','w')
  print('type\tmean\tstd\tmean_early\tstd_early\tmean_mid\tstd_mid\tmean_late\tstd_late', 
        file = master_log)
  
  # all peak timings
  master_df = pd.read_table(args._in, index_col = None)
  
  for f in master_df.trial_type.unique():
    prefix = f
    df = master_df.loc[master_df.trial_type == f,['onset','stage','subj']]
    df_early = df.loc[df.stage=='early',['onset','subj']]
    df_mid = df.loc[df.stage=='mid',['onset','subj']]
    df_late = df.loc[df.stage=='late',['onset','subj']]
    
    # statistic testing of developmental changes
    t_main = np.zeros(3); p_main = np.zeros(3)
    p_main[0],t_main[0] = strat_bs(df_mid,df_early)
    p_main[1],t_main[1] = strat_bs(df_late,df_early)
    p_main[2],t_main[2] = strat_bs(df_late,df_mid)
    
    # statistic testing of inter-group changes
    p_psy = np.zeros(4); t_psy = np.zeros(4);
    p_ord = np.zeros(4); t_ord = np.zeros(4);
    p_hand = np.zeros(4); t_hand = np.zeros(4);
    p_sex = np.zeros(4); t_sex = np.zeros(4);
    
    for i, tmp_df in zip(range(4), [df,df_early,df_mid,df_late]):
      p_psy[i],t_psy[i] = strat_bs(tmp_df.loc[tmp_df.subj.isin(psy_yes),:],
                                   tmp_df.loc[tmp_df.subj.isin(psy_no),:])
      p_ord[i],t_ord[i] = strat_bs(tmp_df.loc[tmp_df.subj.isin(cs2p_first),:],
                                   tmp_df.loc[tmp_df.subj.isin(cs2u_first),:])
      p_hand[i],t_hand[i] = strat_bs(tmp_df.loc[tmp_df.subj.isin(hand_l),:],
                                   tmp_df.loc[tmp_df.subj.isin(hand_r),:])
      p_sex[i], t_sex[i] = strat_bs(tmp_df.loc[tmp_df.subj.isin(male),:],
                                   tmp_df.loc[tmp_df.subj.isin(female),:])
    
    # output p and t-values
    main_df = pd.DataFrame(dict(t_value = t_main, p_value = p_main), 
                           index = ['early v mid','early v late','mid v late'])
    main_df.to_csv(f'{args.out}{prefix}.log', sep = '\t', header = True, index = True)
    
    ctrl_df = pd.DataFrame(
      dict(
        t_psych_background = t_psy,p_psych_background = p_psy,
        t_order = t_ord, p_order = p_ord, t_hand = t_hand, p_hand = p_hand,
        t_sex = t_sex, p_sex = p_sex),
      index = ['total','early','mid','late'])
    ctrl_df.to_csv(f'{args.out}{prefix}_ctrlvars.log', sep = '\t', index = True,
                   header = True)

    # weighted mean
    w_mean = [df.onset.mean(),df_early.onset.mean(),df_mid.onset.mean(),
              df_late.onset.mean()]
    w_std = [df.onset.std(),df_early.onset.std(),df_mid.onset.std(),
              df_late.onset.std()]
    
    # print stats to master logout
    print(f'{prefix}\t{w_mean[0]:.4f}\t{w_std[1]:.4f}\t{w_mean[1]:.4f}\t'+
              f'{w_std[1]:.4f}\t{w_mean[2]:.4f}\t{w_std[2]:.4f}\t{w_mean[3]:.4f}\t{w_std[3]:.4f}', file = master_log)
    
      
    # plot figure
    if fnmatch(f,'*1order*'): c = '#ffc000'
    if fnmatch(f,'*2pair*'): c = '#c00000'
    if fnmatch(f,'*2unp*'): c = '#0070c0'
    # remove total onset data
    w_mean = w_mean[1:]
    w_std = w_std[1:]
    subj_mean = []
    for i in [df_early,df_mid,df_late]: subj_mean.append(i.groupby('subj').mean())
    subj_mean = pd.concat(subj_mean,axis = 'columns').to_numpy()
    
    _,ax = plt.subplots(figsize = (4,4))
    for i in range(subj_mean.shape[0]):
      ax.plot([1,2,3], subj_mean[i,:], marker = '.', color = '#dfdfdf')
    ax.errorbar([1,2,3],w_mean, yerr = w_std, color = c, ecolor = c)
    ax.set_ylim((-550,0))
    ax.set_yticks([-500,-400,-300,-200,-100,0])
    ax.set_xticks([1,2,3], ['early\nacquisition','mid-\nacquisition','late\nacquisition'])
    
    # significance bars
    yref = -100
    yrange = 500
    ax.plot((1,2),(yref+0.1*yrange,yref+0.1*yrange),c = 'k')
    ax.text(1.5,yref+.07*yrange,siglabel(p_main[0]),horizontalalignment='center',
        verticalalignment='center')
    ax.plot((2,3),(yref+.12*yrange,yref+0.12*yrange),c = 'k')
    ax.text(2.5,yref+.09*yrange,siglabel(p_main[2]),horizontalalignment='center',
        verticalalignment='center')
    ax.plot((1,3),(yref+.14*yrange,yref+0.14*yrange),c = 'k')
    ax.text(2,yref+.16*yrange,siglabel(p_main[1]),horizontalalignment='center',
        verticalalignment='center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylabel('Onset timing (ms)')
    plt.savefig(f'{args.out}{prefix}.pdf',bbox_inches = 'tight')
    plt.savefig(f'{args.out}{prefix}.png',bbox_inches = 'tight')
    plt.close()
    
    toc = t() - tic
    print(f'Finished processing {f}. time = {toc:.3f}')
    
if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser(description = 'Cross-subject analysis - stats for eyeblink timing')
  parser.add_argument('-s','--subj', dest = 'subj', default = '../params/subjs.txt',
                      help = 'list of subjs and grouping')
  parser.add_argument('-i','--in', dest = '_in', default = '../stats/onset_hist/all_cr_onset.txt',
                      help = 'input file, contains onset timing for all subjects')
  parser.add_argument('-o','--out',dest = 'out', default = '../stats/onset',
                      help = 'output directory')
  # always overwrites
  args = parser.parse_args()
  main(args)