# -*- coding: utf-8 -*-
#!/usr/bin/env python3
'''
stats_summation.py
This script performs a permutation test for CR across CS2P-CS1 and CS1-US
reconsolidation trials
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
  aware = subjs.loc[subjs.aware == 1, 'subj_id'] - 1
  psy_yes = subjs.loc[subjs.psy == 1, 'subj_id'] - 1
  psy_no = subjs.loc[subjs.psy == 0, 'subj_id'] -1
  cs2p_first = subjs.loc[subjs.order == 'cs2p', 'subj_id']-1
  cs2u_first = subjs.loc[subjs.order == 'cs2u', 'subj_id']-1
  hand_l = subjs.loc[subjs.cs2p_hand == 'LEFT', 'subj_id']-1
  hand_r = subjs.loc[subjs.cs2p_hand == 'RIGHT', 'subj_id']-1
  male = subjs.loc[subjs.sex == 'M', 'subj_id'] -1
  female = subjs.loc[subjs.sex == 'F', 'subj_id'] -1
  
  # the weighted bootstrap function for CRs (paired samples)
  def ttest_w_bs(p1,p2,ntrials, nbs = 10000):
    bs_stat = np.zeros(nbs)
    ntrials = ntrials.astype(np.int32)
    n = len(p1)
    
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
    n1 = n1.astype(np.int32); n2 = n2.astype(np.int32)
    
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
  
  # weighted means and standard deviations
  def wmean_std(data, weight):
    w_mean = (data*weight).sum() / weight.sum()
    w_var = (data**2 *weight).sum() / weight.sum() - w_mean**2
    w_std = (w_var * len(data) / (len(data) - 1))**.5
    return w_mean,w_std
  
  # files to process
  os.chdir(args._in)
  for stat_type in ['cr']:
    f_2p = []; f_cs1 = []
    for f in os.listdir():
      if fnmatch(f,'*all_2pair*') and fnmatch(f,f'*{stat_type}*'): f_2p = f
      # although it is named as all_2pair, CS2P-only trials have been removed in
      # the script subj_analysis.py
      if fnmatch(f,'*rem_1order*') and fnmatch(f,f'*{stat_type}*'): f_cs1 = f
    
    master_log = open(f'{args.out}{stat_type}.log','w')
    
    df_2p = pd.read_table(f_2p, index_col = 0)
    df_cs1 = pd.read_table(f_cs1, index_col = 0)
    
    p_psy = np.zeros(4); t_psy = np.zeros(4);
    p_ord = np.zeros(4); t_ord = np.zeros(4);
    p_hand = np.zeros(4); t_hand = np.zeros(4);
    p_sex = np.zeros(4); t_sex = np.zeros(4);
    
    for stage, i in zip(df_2p.columns[1:],range(4)):
      stat_2p = df_2p[stage].to_numpy()
      stat_cs1 = df_cs1[stage].to_numpy()
      if fnmatch(stage,'*total*'):
        n_2p = df_2p['trial_count'].to_numpy()
        n_cs1 = df_cs1['trial_count'].to_numpy()
      else:
        n_2p = df_2p['trial_count'].to_numpy()/3
        n_cs1 = df_cs1['trial_count'].to_numpy()/3
      
      ntrials = (n_2p * n_cs1) ** 0.5
      # weight by the geometric mean number of trials because the number of trials
      # in paired and CS1 are vastly different
      p_all,t_all = ttest_w_bs(stat_2p,stat_cs1,ntrials)
      p_aware,t_aware = ttest_w_bs(stat_2p[aware],stat_cs1[aware],ntrials[aware])
      
      diff = stat_2p - stat_cs1
      
      # statistic testing of inter-group changes
      p_psy[i],t_psy[i] = ttest_ind_bs(diff[psy_no],diff[psy_yes],ntrials[psy_no],
                                          ntrials[psy_yes])
      p_ord[i],t_ord[i] = ttest_ind_bs(diff[cs2p_first],diff[cs2u_first],
                           ntrials[cs2p_first],ntrials[cs2u_first])
      p_hand[i],t_hand[i] = ttest_ind_bs(diff[hand_l],diff[hand_r],ntrials[hand_l],
                                     ntrials[hand_r])
      p_sex[i], t_sex[i] = ttest_ind_bs(diff[male],diff[female],ntrials[male],
                                     ntrials[female])
      
      # weighted mean
      w_mean_2p, w_std_2p = wmean_std(stat_2p, n_2p)
      w_mean_cs1, w_std_cs1 = wmean_std(stat_cs1, n_cs1)
    
      print(f'Statistics for {stage}', file = master_log)
      print(f'Overall t = {t_all:.3f}, p = {p_all:.4f}', file = master_log)
      print(f'aware subj only t = {t_aware:.3f}, p = {p_aware:.4f}', file = master_log)
      print(f'psy t = {t_psy[i]:.3f}, p = {p_psy[i]:.4f}', file = master_log)
      print(f'order t = {t_ord[i]:.3f}, p = {p_ord[i]:.4f}', file = master_log)
      print(f'hand t = {t_hand[i]:.3f}, p = {p_hand[i]:.4f}', file = master_log)
      print(f'sex t = {t_sex[i]:.4f}, p = {p_sex[i]:.4f}', file = master_log)
      print(f'observed weighted mean for paired: {w_mean_2p:.3f} +/- {w_std_2p:.3f}', file = master_log)
      print(f'observed weighted mean for unpaired: {w_mean_cs1:.3f} +/- {w_std_cs1:.3f}', file = master_log)
      
      # plot figure
      c_2p = '#c00000'; c_cs1 = '#ffc000'
      
      _,ax = plt.subplots(figsize = (3,4))
      for i in range(len(stat_2p)):
        ax.plot([1,2], [stat_2p[i],stat_cs1[i]], marker = '.', color = '#dfdfdf')
      ax.errorbar([1],w_mean_2p, yerr = w_std_2p, color = c_2p, ecolor = c_2p)
      ax.errorbar([2],w_mean_cs1, yerr = w_std_cs1, color = c_cs1, ecolor = c_cs1)
      ax.bar([1],[w_mean_2p], color = c_2p, edgecolor = c_2p, fill = False)
      ax.bar([2],[w_mean_cs1], color = c_cs1, edgecolor = c_cs1, fill = False)
      ax.set_xticks([1,2], ['CS2P-CS1','CS1 only'])
      
      if stat_type == 'cr':
        ax.set_ylim((0,1))
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1],['0','20%','40%','60%','80%','100%'])
        yref = 0.8; yrange = 1
        ax.set_ylabel('CR%')
      else:
        ax.set_ylim((-550,0))
        ax.set_yticks([-500,-400,-300,-200,-100,0])
        yref =-100; yrange = 500
        ax.set_ylabel('Onset time (ms)')
      
      # significance bars
      ax.plot((1,2),(yref+0.1*yrange,yref+0.1*yrange),c = 'k')
      ax.text(1.5,yref+.07*yrange,siglabel(p_all),horizontalalignment='center',
          verticalalignment='center')
      
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
      
      plt.savefig(f'{args.out}{stat_type}_{stage}.pdf',bbox_inches = 'tight')
      plt.savefig(f'{args.out}{stat_type}_{stage}.png',bbox_inches = 'tight')
      plt.close()
      
      # plot same fig for aware only sample
      stat_2pa = stat_2p[aware]; stat_cs1a = stat_cs1[aware]
      w_mean_2pa, w_std_2pa = wmean_std(stat_2pa, ntrials[aware])
      w_mean_cs1a, w_std_cs1a = wmean_std(stat_cs1a, ntrials[aware])
      _,ax = plt.subplots(figsize = (3,4))
      for i in range(len(stat_2pa)):
        ax.plot([1,2], [stat_2pa[i],stat_cs1a[i]], marker = '.', color = '#dfdfdf')
      ax.errorbar([1],w_mean_2pa, yerr = w_std_2pa, color = c_2p, ecolor = c_2p)
      ax.errorbar([2],w_mean_cs1a, yerr = w_std_cs1a, color = c_cs1, ecolor = c_cs1)
      ax.bar([1],[w_mean_2pa], color = c_2p, edgecolor = c_2p, fill = False)
      ax.bar([2],[w_mean_cs1a], color = c_cs1, edgecolor = c_cs1, fill = False)
      ax.set_xticks([1,2], ['CS2P-CS1','CS1 only'])
      
      if stat_type == 'cr':
        ax.set_ylim((0,1))
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1],['0','20%','40%','60%','80%','100%'])
        yref = 0.8; yrange = 1
        ax.set_ylabel('CR%')
      else:
        ax.set_ylim((-550,0))
        ax.set_yticks([-500,-400,-300,-200,-100,0])
        yref =-100; yrange = 500
        ax.set_ylabel('Onset time (ms)')
      
      # significance bars
      ax.plot((1,2),(yref+0.1*yrange,yref+0.1*yrange),c = 'k')
      ax.text(1.5,yref+.07*yrange,siglabel(p_aware),horizontalalignment='center',
          verticalalignment='center')
      
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
      
      plt.savefig(f'{args.out}{stat_type}_{stage}_aware.pdf',bbox_inches = 'tight')
      plt.savefig(f'{args.out}{stat_type}_{stage}_aware.png',bbox_inches = 'tight')
      plt.close()
      
      toc = t() - tic
      print(f'Finished processing {stage} of {stat_type}. time = {toc:.3f}')
    
    ctrl_df = pd.DataFrame(
      dict(
        t_psych_background = t_psy,p_psych_background = p_psy,
        t_order = t_ord, p_order = p_ord, t_hand = t_hand, p_hand = p_hand,
        t_sex = t_sex, p_sex = p_sex),
      index = ['total','early','mid','late'])
    ctrl_df.to_csv(f'{args.out}{stat_type}_ctrlvars.log', sep = '\t', index = True,
                   header = True)
    
    # contrast
    diff = df_2p.iloc[:,2:].to_numpy() - df_cs1.iloc[:,2:].to_numpy()
    p12,t12 = ttest_w_bs(diff[:,0],diff[:,1],ntrials)
    p13,t13 = ttest_w_bs(diff[:,0],diff[:,2],ntrials)
    p23,t23 = ttest_w_bs(diff[:,1],diff[:,2],ntrials)
    w_mean = np.zeros(3); w_std = np.zeros(3)
    for i in range(3): w_mean[i],w_std[i] = wmean_std(diff[:,i],ntrials)
    print('Development of CR2P-CR2U differences', file = master_log)
    print(f'1/2 t = {t12:.4f}, p = {p12:.4f}', file = master_log)
    print(f'1/3 t = {t13:.4f}, p = {p13:.4f}', file = master_log)
    print(f'2/3 t = {p13:.4f}, p = {p23:.4f}', file = master_log)
    
    _,ax = plt.subplots(figsize = (4,4))
    for i in range(diff.shape[0]):
      ax.plot([1,2,3], diff[i,:], marker = '.', color = '#dfdfdf')
    ax.errorbar([1,2,3],w_mean, yerr = w_std, color = 'k', ecolor = 'k')
    ax.set_ylim((-1,1))
    ax.set_yticks([-1,-0.5,0,0.5,1])
    ax.set_xticks([1,2,3], ['early\nacquisition','mid-\nacquisition',
                            'late\nacquisition'])
    
    # significance bars
    yref = 0.6
    yrange = 2
    ax.plot((1,2),(yref+0.1*yrange,yref+0.1*yrange),c = 'k')
    ax.text(1.5,yref+.07*yrange,siglabel(p12),horizontalalignment='center',
        verticalalignment='center')
    ax.plot((2,3),(yref+.12*yrange,yref+0.12*yrange),c = 'k')
    ax.text(2.5,yref+.09*yrange,siglabel(p23),horizontalalignment='center',
        verticalalignment='center')
    ax.plot((1,3),(yref+.14*yrange,yref+0.14*yrange),c = 'k')
    ax.text(2,yref+.16*yrange,siglabel(p13),horizontalalignment='center',
        verticalalignment='center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylabel('CR% (compound) - CR1%')
    plt.savefig(f'{args.out}{stat_type}_difference.pdf',bbox_inches = 'tight')
    plt.savefig(f'{args.out}{stat_type}_difference.png',bbox_inches = 'tight')
    plt.close()
    
if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser(description = 'Cross-subject analysis - stats for CR%')
  parser.add_argument('-s','--subj', dest = 'subj', default = '../params/subjs.txt',
                      help = 'list of subjs and grouping')
  parser.add_argument('-i','--in', dest = '_in', default = '../stats/summary',
                      help = 'input directory')
  parser.add_argument('-o','--out',dest = 'out', default = '../stats/summation',
                      help = 'output directory')
  # always overwrites
  args = parser.parse_args()
  main(args)