# -*- coding: utf-8 -*-
#!/usr/bin/env python3
'''
stats_ttt.py
This script performs a permutation test for trials to threshold across stages of the experiment
'''

def main(args):
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import os
  from fnmatch import fnmatch
  from time import perf_counter as t
  
  rng = np.random.Generator(np.random.MT19937(310110))
  tic = t()
  
  # fail-safe design
  args._in = os.path.realpath(args._in) + '/'
  args.out = os.path.realpath(args.out) + '/'
  
  if not os.path.isdir(args.out): os.mkdir(args.out)
  
  # subjects list
  subjs = pd.read_table(args.subj,index_col = None).sort_values(by = 'subj_id')
  
  # subjects grouping
  psy_yes = subjs.loc[subjs.psy == 1, 'subj_id'] - 1
  psy_no = subjs.loc[subjs.psy == 0, 'subj_id'] -1
  cs2p_first = subjs.loc[subjs.order == 'cs2p', 'subj_id']-1
  cs2u_first = subjs.loc[subjs.order == 'cs2u', 'subj_id']-1
  hand_l = subjs.loc[subjs.cs2p_hand == 'LEFT', 'subj_id']-1
  hand_r = subjs.loc[subjs.cs2p_hand == 'RIGHT', 'subj_id']-1
  male = subjs.loc[subjs.sex == 'M', 'subj_id'] -1
  female = subjs.loc[subjs.sex == 'F', 'subj_id'] -1
  
  # files to process
  os.chdir(args._in)
  flist = []
  for f in os.listdir():
    if fnmatch(f,'*ttt*'): flist.append(f)
  
  # weighted independent samples bootstrap t-test across groups
  def ttest_ind_bs(p1,p2,n1 = None,n2 = None, nbs = 100000):
    p1 = p1[~np.isnan(p1)]; p2 = p2[~np.isnan(p2)]
    if min((len(p1),len(p2))) == 0: return np.nan,np.nan,np.nan
    if type(n1) == type(None): n1 = np.ones(len(p1))
    if type(n2) == type(None): n2 = np.ones(len(p2))
    sqr_nbs = int(nbs**.5)
    nbs = sqr_nbs ** 2 # must be an integer
    
    # null hypothesis
    p_null = ((p1*n1).sum() + (p2*n2).sum())/(n1.sum()+n2.sum())
    
    # observed t-value
    obs_mean1 = (p1*n1).sum()/n1.sum(); obs_mean2 = (p2*n2).sum()/n2.sum() # delta p
    obs_var1 = ((p1**2*n1).sum()/n1.sum() - obs_mean1**2) * len(n1)/(len(n1)-1)
    obs_var2 = ((p2**2*n2).sum()/n2.sum() - obs_mean2**2) * len(n2)/(len(n2)-1)
    obs_stat = (obs_mean1 - obs_mean2)/(obs_var1 + obs_var2)**0.5 * \
      (len(n1)+len(n2) - 2)**0.5 / (1/len(n1) + 1/len(n2)) ** 0.5
    
    # transform data
    p1_t = p1 - p_null; p2_t = p2 - p_null
    
    # bootstrap for each independent sample
    p1_bs = np.zeros((sqr_nbs,len(n1))); p2_bs = np.zeros((sqr_nbs,len(n2)))
    bs_stat = np.zeros((sqr_nbs,sqr_nbs))
    for i in range(sqr_nbs):
      p1_bs[i,:] = rng.choice(p1_t,size = len(n1))
      p2_bs[i,:] = rng.choice(p2_t,size = len(n2))
    
    for i in range(sqr_nbs):
      for j in range(sqr_nbs):
        q1 = p1_bs[i,:]; q2 = p2_bs[j,:]
        bs_mean1 = (q1*n1).sum()/n1.sum(); bs_mean2 = (q2*n2).sum()/n2.sum() # delta q
        bs_var1 = ((q1**2*n1).sum()/n1.sum() - bs_mean1**2) * len(n1)/(len(n1)-1)
        bs_var2 = ((q2**2*n2).sum()/n2.sum() - bs_mean2**2) * len(n2)/(len(n2)-1)
        if bs_var1 < 0: bs_var1 = 10**-10 # this is to correct for float errors
        if bs_var2 < 0: bs_var2 = 10**-10 # this is to correct for float errors
        bs_stat[i,j] = (bs_mean1 - bs_mean2)/(bs_var1 + bs_var2)**0.5 * \
          (len(n1)+len(n2) - 2)**0.5 / (1/len(n1) + 1/len(n2)) ** 0.5
      
    bs_stat = bs_stat.reshape(nbs)
    p = sum(bs_stat < obs_stat)
    p = min((p, nbs - p))
    p /= (nbs/2) # 2-tailed p-value
    
    return p, None, obs_stat
  
  # convert CI level to significance labels
  def siglabel(p):
    if p > 0.1: return 'ns'
    elif p > 0.05: return f'p = {p:.3f}'
    elif p > 0.01: return '*'
    elif p > 10**-3: return '**'
    elif p > 10**-4: return '***'
    else: return '****'
  
  # overall p-values log
  master_log = open(f'{args.out}all_pvalues.log','w')
  print('\t\tThreshold\t\tMax\t', file = master_log)
  print('type\tpsy\torder\thand\tpsy\torder\thand', file = master_log)
  
  for f in flist:
    print(f'Processing {f}.')
    prefix = f.replace('.txt','')
    df = pd.read_table(f, index_col = 0)
        
    # extract data
    tt4u = df['time_to_th_univ'].to_numpy()
    ttmu = df['time_to_max_univ'].to_numpy()

    # statistic testing of inter-group changes
    p_psy,_,obs_psy = ttest_ind_bs(tt4u[psy_no],tt4u[psy_yes])
    p_ord,_,obs_ord = ttest_ind_bs(tt4u[cs2p_first],tt4u[cs2u_first])
    p_hand,_,obs_hand = ttest_ind_bs(tt4u[hand_l],tt4u[hand_r])
    p_sex,_, obs_sex = ttest_ind_bs(tt4u[male],tt4u[female])
    
    p_psy_m,_,obs_psy_m = ttest_ind_bs(ttmu[psy_no],ttmu[psy_yes])
    p_ord_m,_,obs_ord_m = ttest_ind_bs(ttmu[cs2p_first],ttmu[cs2u_first])
    p_hand_m,_,obs_hand_m = ttest_ind_bs(ttmu[hand_l],ttmu[hand_r])
    p_sex_m,_, obs_sex_m = ttest_ind_bs(tt4u[male],tt4u[female])
    
    with open(f'{args.out}{prefix}.log','w') as logout:
      print('Trials to threshold', file = logout)
      print(f'psy t = {obs_psy:.4f}, p = {p_psy:.4f}', file = logout)
      print(f'order t = {obs_ord:.4f}, p = {p_ord:.4f}', file = logout)
      print(f'hand t = {obs_hand:.4f}, p = {p_hand:.4f}', file = logout)
      print(f'sex t = {obs_sex:.4f}, p = {p_sex:.4f}', file = logout)
      print('Trials to max', file = logout)
      print(f'psy t = {obs_psy_m:.4f}, p = {p_psy_m:.4f}', file = logout)
      print(f'order t = {obs_ord_m:.4f}, p = {p_ord_m:.4f}', file = logout)
      print(f'hand t = {obs_hand_m:.4f}, p = {p_hand_m:.4f}', file = logout)
      print(f'sex t = {obs_sex_m:.4f}, p = {p_sex_m:.4f}', file = logout)
      
    # print stats to master logout
    print(f'{prefix}\t{p_psy:.4f}\t{p_ord:.4f}\t{p_hand:.4f}\t'+
              f'{p_psy_m:.4f}\t{p_ord_m:.4f}\t{p_hand_m:.4f}', file = master_log)
    
    # plot figure
    if fnmatch(f,'*1order*'): c = '#ffc000'
    if fnmatch(f,'*2pair*'): c = '#c00000'
    if fnmatch(f,'*2unp*'): c = '#0070c0'
    
    _,ax = plt.subplots(figsize = (1.5,4))
    for i in range(len(tt4u)):
      # jitter
      ax.plot([1+rng.normal(scale=.05)],tt4u[i], marker = 'o', color = '#dfdfdf')
      ax.plot([2+rng.normal(scale=.05)],ttmu[i], marker = 'o', color = '#dfdfdf') 
    ax.errorbar([1,2],[tt4u.mean(),ttmu.mean()], yerr = [tt4u.std(),ttmu.std()], 
                color = c, ecolor = c,linestyle = 'none')
    ax.bar([1,2],[tt4u.mean(),ttmu.mean()], color = c, edgecolor = c, fill = False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([1,2],['trials\nto\nthres.','trials\nto\nmax'])
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
  parser.add_argument('-i','--in', dest = '_in', default = '../stats/summary',
                      help = 'input directory')
  parser.add_argument('-o','--out',dest = 'out', default = '../stats/ttt',
                      help = 'output directory')
  # always overwrites
  args = parser.parse_args()
  main(args)