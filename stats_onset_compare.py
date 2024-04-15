# -*- coding: utf-8 -*-
#!/usr/bin/env python3

'''
stats_onset_compare.py
This script tests onset timing across different stages by trial-wise data
'''

def main(args):
  import os
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  from fnmatch import fnmatch
  from time import perf_counter as t
  
  rng = np.random.Generator(np.random.MT19937(310110))
  tic = t()
  
  # process trial types
  def reg_name(s):
    if fnmatch(s,'*1*'): return 'rem_1order','CS1','#ffc000'
    if fnmatch(s,'*2p*'): return 'pr_2pair','CS2P','#c00000'
    if fnmatch(s,'*2u*'): return 'pr_2unp','CS2U','#0070c0'
  type1,label1,c1 = reg_name(args._type[0])
  type2,label2,c2 = reg_name(args._type[1])
  
  # IO fail-safe design
  _in = os.path.realpath(args._in)
  out = os.path.realpath(args.out) + f'/{label1}_{label2}'
  if not os.path.isdir(out): os.mkdir(out)
  os.chdir(out)
  
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
  
  # convert p-values to significance labels
  def siglabel(p):
    if p > 0.1: return 'ns'
    elif p > 0.05: return f'p = {p:.3f}'
    elif p > 0.01: return '*'
    elif p > 10**-3: return '**'
    elif p > 10**-4: return '***'
    else: return '****'
    
  # def the bar plot function
  def bar_subj(data1,data2,t1,t2,p,out_fname):
    _,label1,c1 = reg_name(t1)
    _,label2,c2 = reg_name(t2)
    d1 = data1.iloc[:,0].to_numpy(); g1 = data1.iloc[:,1].to_numpy()
    d2 = data2.iloc[:,0].to_numpy(); g2 = data2.iloc[:,1].to_numpy()
    
    _,ax = plt.subplots(figsize=(3,4))
    for i in np.unique(np.concatenate((g1,g2))): # each subject
      try: m1 = d1[g1==i].mean()
      except: m1=np.nan
      try: m2 = d2[g2==i].mean()
      except: m2 = np.nan
      ax.plot([1,2], [m1,m2], marker = '.', color = '#dfdfdf')
    ax.errorbar([1],d1.mean(), yerr = d1.std(), color = c1, ecolor = c1)
    ax.errorbar([2],d2.mean(), yerr = d2.std(), color = c2, ecolor = c2)
    ax.bar([1],[d1.mean()], color = c1, edgecolor = c1, fill = False)
    ax.bar([2],[d2.mean()], color = c2, edgecolor = c2, fill = False)
    ax.set_xticks([1,2], [label1,label2])
    ax.set_ylim((-550,0))
    ax.set_yticks([-500,-400,-300,-200,-100,0])
    yref = -450; yrange = 550
    ax.set_ylabel('Onset time (ms)')
    
    # significance bars
    ax.plot((1,2),(yref-0.1*yrange,yref-0.1*yrange),c = 'k')
    ax.text(1.5,yref-.13*yrange,siglabel(p),horizontalalignment='center',
        verticalalignment='center')
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.savefig(out_fname,bbox_inches = 'tight')
    plt.savefig(out_fname.replace('pdf','png'))
    plt.close()
    return
  
  # input data
  data = pd.read_table(_in)
  data1 = data.loc[data.trial_type == type1,:]
  data2 = data.loc[data.trial_type == type2,:]
  p_values = np.zeros(4); t_values = np.zeros(4)
  p_values[0],t_values[0] = strat_bs(data1[['onset','subj']],data2[['onset','subj']])
  bar_subj(data1[['onset','subj']],data2[['onset','subj']],type1,type2,p_values[0],
           f'{out}/{label1}_{label2}_onset.pdf')
  toc = t() - tic
  print(f'Processed data across acquisition stages. time = {toc:.3f}')
  for i,j in zip(range(1,4),['early','mid','late']):
    tmp1 = data1.loc[data1.stage==j,['onset','subj']]
    tmp2 = data2.loc[data2.stage==j,['onset','subj']]
    p_values[i],t_values[i] = strat_bs(tmp1,tmp2)
    bar_subj(tmp1,tmp2,type1,type2,p_values[i],f'{out}/{label1}_{label2}_{j}_onset.pdf')
    toc = t() - tic
    print(f'Processed {j} stage of acquisition. time = {toc:.3f}')
  # output log
  out_df = pd.DataFrame(dict(t = t_values, p = p_values), 
                        index = ['total','early','mid','late'])
  out_df.to_csv(f'{out}/{label1}_{label2}_onset.log', sep = '\t')

if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser(description = 'Cross-subject analysis - stats for eyeblink timing, by trial-wise data')
  parser.add_argument('_type', help = 'type of trials to analyse', nargs=2)
  parser.add_argument('-s','--subj', dest = 'subj', default = '../params/subjs.txt',
                      help = 'list of subjs and grouping')
  parser.add_argument('-i','--in', dest = '_in', default = '../stats/onset_hist/all_cr_onset.txt',
                      help = 'input file, contains onset timing for all subjects')
  parser.add_argument('-o','--out',dest = 'out', default = '../stats/',
                      help = 'output directory')
  # always overwrites
  args = parser.parse_args()
  main(args)