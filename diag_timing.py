# -*- coding: utf-8 -*-
#!/usr/bin/env python3
'''
diag_timing.py
This script plots the distribution of three timing measures
'''

def main(args):
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  import os
  from fnmatch import fnmatch
  
  print(f'Processing subject {args.subj}')
  
  # fail-safe design
  args._in = os.path.realpath(args._in) + '/'
  args.out = os.path.realpath(args.out) + '/'
  
  if not os.path.isdir(args.out): os.mkdir(args.out)
  os.chdir(args.out)
  if not os.path.isdir('timing'): os.mkdir('timing')
  
  if os.path.isfile(f'{args.out}timing/{args.subj}.png') and not args.force:
    return
  
  os.chdir(args._in);
  
  if args.subj != 'all':
    os.chdir(args.subj)
    flist = []
    for f in os.listdir():
      if fnmatch(f,'*peakinfo*') and not fnmatch(f,'*all_*'): flist.append(f)
  else:
    flist = []
    for d in os.listdir():
      os.chdir(d)
      for f in os.listdir():
        if fnmatch(f,'*peakinfo*') and not fnmatch(f,'*all_*'): 
          flist.append(os.path.realpath(f))
      os.chdir(args._in)
  
  dflist = []
  for f in flist:
    df = pd.read_csv(f)
    if fnmatch(f,'*1order*'): df['type'] = 'CS1'
    if fnmatch(f,'*ext*'): df['type'] = 'extinction'
    elif fnmatch(f,'*2pair*'): df['type'] = 'CS2P'
    elif fnmatch(f,'*2unp*'): df['type'] = 'CS2U'
    dflist.append(df)
  all_trials = pd.concat(dflist,axis = 0).reset_index(drop = True)
  
  corrmat = all_trials[['peak','left_d2','left_2d1','left_trough']].corr().to_numpy()
   
  _,ax = plt.subplots(2,3,figsize = (21,14))
  sns.regplot(all_trials,x='left_d2',y='left_2d1',ax = ax[0,0])
  sns.scatterplot(all_trials,x='left_d2',y='left_2d1',hue = 'type',ax = ax[0,0])
  sns.regplot(all_trials,x='left_d2',y='left_trough',ax = ax[0,1])
  sns.scatterplot(all_trials,x='left_d2',y='left_trough',hue = 'type',ax = ax[0,1])
  sns.regplot(all_trials,x='left_trough',y='left_2d1',ax = ax[0,2])
  sns.scatterplot(all_trials,x='left_trough',y='left_2d1',hue = 'type',ax = ax[0,2])
  
  sns.regplot(all_trials,x='peak',y='left_d2',ax = ax[1,0])
  sns.scatterplot(all_trials,x='peak',y='left_d2',hue = 'type',ax = ax[1,0])
  sns.regplot(all_trials,x='peak',y='left_2d1',ax = ax[1,1])
  sns.scatterplot(all_trials,x='peak',y='left_2d1',hue = 'type',ax = ax[1,1])
  sns.regplot(all_trials,x='peak',y='left_trough',ax = ax[1,2])
  sns.scatterplot(all_trials,x='peak',y='left_trough',hue = 'type',ax = ax[1,2])
  
  ax[0,0].set_title(f'r = {corrmat[1,2]:.3f}')
  ax[0,1].set_title(f'r = {corrmat[1,3]:.3f}')
  ax[0,2].set_title(f'r = {corrmat[2,3]:.3f}')
  ax[1,0].set_title(f'r = {corrmat[0,1]:.3f}')
  ax[1,1].set_title(f'r = {corrmat[0,2]:.3f}')
  ax[1,2].set_title(f'r = {corrmat[0,3]:.3f}')
  
  plt.savefig(f'{args.out}timing/{args.subj}.png')
  plt.savefig(f'{args.out}timing/{args.subj}.pdf', bbox_inches = 'tight')
  plt.close()

if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser(description = 'Stats for different timing measures for a subj')
  parser.add_argument('subj', help = 'subject ID')
  parser.add_argument('-i','--in', dest = '_in', default = '../score',
                      help = 'input directory')
  parser.add_argument('-o','--out',dest = 'out', default = '../stats',
                      help = 'output directory')
  parser.add_argument('-f','--force', action = 'store_true', default = False, 
                      help = 'force overwrite')
  args = parser.parse_args()
  main(args)