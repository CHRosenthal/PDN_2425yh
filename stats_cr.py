# -*- coding: utf-8 -*-
#!/usr/bin/env python3
'''
stats_cr.py
This script performs a permutation test for CR across stages of the experiment
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
    if fnmatch(f,'*cr*') and not fnmatch(f,'total_cr*'): flist.append(f)
  
  # the weighted bootstrap function for CRs (paired samples)
  def ttest_w_bs(p1,p2,ntrials, nbs = 10000):
    bs_stat = np.zeros(nbs)
    n = len(p1)
    
    # for i in range(nbs):
    #   bs_sample = 0
    #   for j in range(n):
    #     # for each subject, bootstrap resample trial_count/3 trials for that subject
    #     # for the two stages of the experiment, and estimate the difference
    #     # this is to satisfy the 'paired samples' condition
    #     bs_sample += rng.binomial(ntrials[j],p2[j])
    #     bs_sample -= rng.binomial(ntrials[j],p1[j])
    #   # the resampled difference in CR% is calculated as below
    #   bs_stat[i] = bs_sample/ntrials.sum()
    
    # rej = sum(bs_stat < 0)
    # rej = min((rej,nbs - rej))
    # rej /= (nbs/2) # two-tailed max CI range (NOT p value!)
    
    # bs_stat.sort()
    # ci = [bs_stat[int(nbs*.025)],bs_stat[-int(nbs*.025)]]
    # obs = ((p2-p1) * ntrials).sum() / ntrials.sum()
    obs_mean = ((p2-p1) * ntrials).sum() / ntrials.sum()
    obs_var = ((p2-p1)**2 * ntrials).sum() / ntrials.sum() - obs_mean**2
    obs_std = (obs_var*n/(n-1))**.5
    obs_stat = obs_mean/obs_std * n**.5
    
    data_t = p2-p1 - (p2-p1).mean()
    for i in range(nbs):
      bs_sample = rng.choice(data_t, size = n) # only weight once, in the subsequent step
      bs_mean = (bs_sample * ntrials).sum() / ntrials.sum()
      bs_var = (bs_sample**2 * ntrials).sum() / ntrials.sum() - bs_mean**2
      if bs_var < 0: bs_var = 10**-10 # this is to correct for float errors
      bs_std = (bs_var*n/(n-1))**.5
      bs_stat[i] = bs_mean/bs_std * n**.5
    
    p = sum(bs_stat < obs_stat)
    p = min((p, nbs - p))
    p /= (nbs/2) # 2-tailed p-value
    
    return p, obs_stat
  
  # weighted independent samples bootstrap t-test for CRs across groups
  def ttest_ind_bs(p1,p2,n1,n2, nbs = 65536):
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
    
    return p, obs_stat
  
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
  
  for f in flist:
    prefix = f.replace('.txt','')
    df = pd.read_table(f, index_col = 0)
        
    # extract data
    ntrials = df['trial_count'].to_numpy()
    df['diff12'] = df['second_third_cr'] - df['first_third_cr']
    df['diff23'] = df['last_third_cr'] - df['second_third_cr']
    data = df[['total_cr','first_third_cr','second_third_cr','last_third_cr',
            'diff12','diff23']].to_numpy()
    
    # statistic testing of developmental changes
    t_main = np.zeros(3); p_main = np.zeros(3)
    p_main[0],t_main[0] = ttest_w_bs(data[:,2],data[:,1],ntrials/3)
    p_main[1],t_main[1] = ttest_w_bs(data[:,3],data[:,1],ntrials/3)
    p_main[2],t_main[2] = ttest_w_bs(data[:,3],data[:,2],ntrials/3)
    
    # statistic testing of inter-group changes
    p_psy = np.zeros(6); t_psy = np.zeros(6);
    p_ord = np.zeros(6); t_ord = np.zeros(6);
    p_hand = np.zeros(6); t_hand = np.zeros(6);
    p_sex = np.zeros(6); t_sex = np.zeros(6);
    
    for i in range(6):
      p_psy[i],t_psy[i] = ttest_ind_bs(data[psy_no,i],data[psy_yes,i],
                                       ntrials[psy_no],ntrials[psy_yes])
      p_ord[i],t_ord[i] = ttest_ind_bs(data[cs2p_first,i],data[cs2u_first,i],
                           ntrials[cs2p_first],ntrials[cs2u_first])
      p_hand[i],t_hand[i] = ttest_ind_bs(data[hand_l,i],data[hand_r,i],
                                              ntrials[hand_l],ntrials[hand_r])
      p_sex[i], t_sex[i] = ttest_ind_bs(data[male,i],data[female,i],
                                           ntrials[male],ntrials[female])
      if i == 0: ntrials /= 3
    ntrials *= 3
    
    main_df = pd.DataFrame(dict(t_value = t_main, p_value = p_main), 
                           index = ['early v mid','early v late','mid v late'])
    main_df.to_csv(f'{args.out}{prefix}.log', sep = '\t', header = True, index = True)
    
    ctrl_df = pd.DataFrame(
      dict(
        t_psych_background = t_psy,p_psych_background = p_psy,
        t_order = t_ord, p_order = p_ord, t_hand = t_hand, p_hand = p_hand,
        t_sex = t_sex, p_sex = p_sex),
      index = ['total','early','mid','late','mid - early', 'late - mid'])
    ctrl_df.to_csv(f'{args.out}{prefix}_ctrlvars.log', sep = '\t', index = True,
                   header = True)
    
    # weighted means and standard deviations
    def wmean_std(data, weight):
      w_mean = (data*weight).sum() / weight.sum()
      w_var = (data**2 *weight).sum() / weight.sum() - w_mean**2
      w_std = (w_var * len(data) / (len(data) - 1))**.5
      return w_mean,w_std
    
    data = df[['total_cr','first_third_cr','second_third_cr','last_third_cr']].to_numpy()
    w_mean = np.zeros(4)
    w_std = np.zeros(4)
    for i in range(4):
      w_mean[i],w_std[i] = wmean_std(data[:,i],ntrials)
      if i == 0: ntrials/=3
    ntrials *= 3
    # print stats to master logout
    print(f'{prefix}\t{w_mean[0]:.4f}\t{w_std[0]:.4f}\t{w_mean[1]:.4f}\t{w_std[1]:.4f}'+
              f'\t{w_mean[2]:.4f}\t{w_std[2]:.4f}\t{w_mean[3]:.4f}\t{w_std[3]:.4f}', file = master_log)

    # plot figure
    if fnmatch(f,'*1order*'): c = '#ffc000'
    if fnmatch(f,'*2pair*'): c = '#c00000'
    if fnmatch(f,'*2unp*'): c = '#0070c0'
    # remove total CR data
    w_mean = w_mean[1:]
    w_std = w_std[1:]
    data = data[:,1:]
    
    _,ax = plt.subplots(figsize = (4,4))
    for i in range(data.shape[0]):
      if fnmatch(f,'*ext*') and i in [1,3,4]:
        ax.plot([1,2,3], data[i,:], marker = '.', color = 'k')
      else:
        ax.plot([1,2,3], data[i,:], marker = '.', color = '#dfdfdf')
    ax.errorbar([1,2,3],w_mean, yerr = w_std, color = c, ecolor = c)
    ax.set_ylim((-.02,1))
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1],['0','20%','40%','60%','80%','100%'])
    if fnmatch(f,'*ext*'):
      ax.set_xticks([1,2,3], ['early\nextinction','mid-\nextinction',
                              'late\nextinction'])
    else:
      ax.set_xticks([1,2,3], ['early\nacquisition','mid-\nacquisition',
                            'late\nacquisition'])
    
    # significance bars
    yref = .8
    yrange = 1
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
    plt.ylabel('CR%')
    plt.savefig(f'{args.out}{prefix}.pdf',bbox_inches = 'tight')
    plt.savefig(f'{args.out}{prefix}.png',bbox_inches = 'tight')
    plt.close()
    
    toc = t() - tic
    print(f'Finished processing {f}. time = {toc:.3f}')
    
if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser(description = 'Cross-subject analysis - stats for CR%')
  parser.add_argument('-s','--subj', dest = 'subj', default = '../params/subjs.txt',
                      help = 'list of subjs and grouping')
  parser.add_argument('-i','--in', dest = '_in', default = '../stats/summary',
                      help = 'input directory')
  parser.add_argument('-o','--out',dest = 'out', default = '../stats/cr',
                      help = 'output directory')
  # always overwrites
  args = parser.parse_args()
  main(args)