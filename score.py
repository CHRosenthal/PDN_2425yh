# -*- coding: utf-8 -*-
#!/usr/bin/env python3

'''
score.py
This script scores single-trial responses
'''

def find_peaks(ts,cutoff, t_valid, width = 800):
  import numpy as np
  
  hw = int(width/2) 
  # half width; deliberately set larger because peak is not always in the middle
  peaks = []; l_trough = []; r_trough = []; height = []; height_l = []; height_r = []; auc = [];
  l_maxd1 = []; r_maxd1 = []; l_maxd2 = []; r_maxd2 = []; l_2xmaxd1 = []; r_2xmaxd1 = []
  fwhm = []
  ts_copy = ts.copy()
  diff = ts_copy[1:]-ts_copy[:-1] # differential / slope
  diff2 = diff[1:]-diff[:-1] # 2nd differential
  diff2 = np.concatenate([[diff2[0]],diff2]) # add 1ms offset to align w/ original ts
  
  while np.nanmax(ts_copy) - np.nanmin(ts_copy) > cutoff: # if max - min > cutoff, there might be peaks
    tmppeak = np.nanargmax(ts_copy) # peak timing
    
    # coarse cutting of the peaks
    tmplarm = ts_copy[max(tmppeak-hw,0):tmppeak+1][::-1]
    try:
      tmpleft = tmppeak - np.nanargmin(tmplarm.round(3)) # identifies LOCATION
    except: tmpleft = tmppeak
    tmprarm = ts_copy[tmppeak:min(tmppeak+hw, len(ts_copy))]
    try:
      tmpright = tmppeak + np.nanargmin(tmprarm.round(3)) # identifies LOCATION
    except: tmpright = tmppeak
    
    # peak/trough values
    tmpmax = ts_copy[tmppeak]; tmplmin = ts_copy[tmpleft]; tmprmin = ts_copy[tmpright]
    tmph_l = tmpmax - tmplmin; tmph_r = tmpmax - tmprmin
    tmph = max([tmph_l,tmph_r])
    tmph2 = min([tmph_l,tmph_r])
    
    # fine-tuning of timing based on 1st differential
    if tmppeak - tmpleft > 0:
      tmpl_d1 = diff[tmpleft:tmppeak].argmax() + tmpleft; tmpl_2d1 = tmpl_d1 * 2 - tmppeak
      tmpl_d2 = diff2[tmpleft:tmppeak].argmax() + tmpleft
    else:
      tmpl_d1 = tmpleft; tmpl_2d1 = tmpleft; tmpl_d2 = tmpleft
    
    # re-set timing for half-peaks
    # if for the period [tmpleft,t_valid[1]] the difference is < cutoff,
    # tmpl_d2 will be retained as a t>0 value and thus scored invalid in score() fcn
    if tmpleft < t_valid[1] and tmpl_d2 > t_valid[1] and \
      ts_copy[tmpleft:t_valid[1]].max() - ts_copy[tmpleft:t_valid[1]].min() > cutoff:
      tmpl_d1 = diff[tmpleft:t_valid[1]].argmax() + tmpleft;
      tmpl_2d1 = tmpl_d1 * 2 - tmppeak
      tmpl_d2 = diff2[tmpleft:t_valid[1]].argmax() + tmpleft
    
    if tmpright - tmppeak > 0:
      tmpr_d1 = diff[tmppeak:tmpright].argmin() + tmppeak; tmpr_2d1 = tmpr_d1 * 2 - tmppeak
      tmpr_d2 = diff2[tmppeak:tmpright].argmax() + tmppeak
    else:
      tmpr_d1 = tmpright; tmpr_2d1 = tmpright; tmpr_d2 = tmpright
    
    if tmph > cutoff and tmph2 > cutoff / 2: 
      # peaks may be asymmetric
      # this includes half-peaks towards the end of the ts
      
      # crude timing
      peaks.append(tmppeak); l_trough.append(tmpleft); r_trough.append(tmpright)
      
      # heights
      height.append(tmph); height_l.append(tmph_l); height_r.append(tmph_r)
      
      # fine timing
      l_maxd1.append(tmpl_d1); r_maxd1.append(tmpr_d1)
      l_2xmaxd1.append(tmpl_2d1); r_2xmaxd1.append(tmpr_2d1)
      l_maxd2.append(tmpl_d2); r_maxd2.append(tmpr_d2)
      
      # area under curve
      # auc.append(sum(ts_copy[tmpleft:tmpright]) -
      #            min(tmplmin,tmprmin)*(-tmpleft+tmpright))
      auc.append(sum(ts_copy[tmpl_d2:tmpr_d2]) -
                 min(tmplmin,tmprmin)*(tmpr_d2-tmpl_d2))
      
      # FWHM value
      hm = tmpmax - tmph2 / 2
      tmpts = ts_copy[tmpleft:tmpright]
      fwhm.append(tmpts[tmpts > hm].size)
    
    ts_copy[tmpleft:tmpright+1] = np.nan
    # if any(np.isnan(ts_copy[-15:])): ts_copy[-15:] = np.nan # clip off the right tail
    if all(np.isnan(ts_copy)): break
    
  peaks = np.array(peaks)
  props = dict(peak = peaks,
               left_trough = np.array(l_trough), right_trough = np.array(r_trough),
               left_2d1 = np.array(l_2xmaxd1), right_2d1 = np.array(r_2xmaxd1),
               left_d2 = np.array(l_maxd2), right_d2 = np.array(r_maxd2),
               peak_heights = np.array(height), 
               height_l = np.array(height_l), height_r = np.array(height_r),
               auc = np.array(auc), fwhm = fwhm)
  
  return peaks, props

def score(trial,cutoff):
  import numpy as np
  import pandas as pd
  # baseline: [mean, sd, min, max]
  
  cut_start = 500 # -1s to 0.5s, so that peaks w/ start time before -0.5s can be excluded
  t_valid = [1000-cut_start,1550-cut_start] # change this param for change of timing window
  t = trial[cut_start:] # cut version of this trial

  peaks, props = find_peaks(t, cutoff = cutoff, t_valid = t_valid)
  peaks = np.array(peaks)
  props = pd.DataFrame(props)
  peak_start = props['left_d2']
  
  # binary (if there is a peak)
  # find peaks within the above valid time window
  valid_filter = np.zeros(peaks.size).astype('?') # defaults to false
  for i in range(peaks.size):
    if peak_start[i] > t_valid[0] and peak_start[i] < t_valid[1]: valid_filter[i] = True
  valid_peaks = peaks[valid_filter]
  valid_props = props.loc[valid_filter,:].reset_index(drop = True)
  peak_presence = len(valid_peaks) > 0
  
  # update heights and auc for half peaks, to dissociate from signals induced by US  
  for i in range(valid_peaks.size):
    if valid_peaks[i] > t_valid[1] and valid_props['left_d2'][i] < t_valid[1]:
      valid_props.loc[i,'peak_heights'] = t[valid_props['left_trough'][i]:t_valid[1]].max() - \
        t[valid_props['left_trough'][i]]
      valid_props.loc[i,'auc'] = t[valid_props['left_d2'][i]:t_valid[1]].sum() - \
        t[valid_props['left_trough'][i]]*(t_valid[1]-valid_props['left_d2'][i])
  
  # max peak height
  if peak_presence > 0:
    max_height = valid_props['peak_heights'].max()
  else:
    max_height = np.nan
  
  # area under curve
  if peak_presence > 0:
    auc = valid_props['auc'].sum()
  else:
    auc = np.nan
    
  # range
  delta = t[t_valid[0]:t_valid[1]].max() - t[t_valid[0]:t_valid[1]].min()
  
  # start of peaks
  if peak_presence > 0:
    start = valid_props['left_d2'].min()-t_valid[1]
  else: start = np.nan
  s = np.array([peak_presence, max_height, auc, delta, start])
  
  # re_sets the timing to align with US
  for var in ['peak','left_trough','right_trough','left_2d1','right_2d1','left_d2','right_d2']:
    valid_props[var] -= t_valid[1]
  
  return s, valid_props

def main(args):
  print(f'Processing file: {args._in}')
  
  import os
  if os.path.isfile(args.out) and not args.force: return
  import numpy as np
  import pandas as pd
  # import scipy.signal as sgn
  from time import perf_counter as t
  
  tic = t()
  
  x = np.loadtxt(args._in, delimiter = '\t')
  # each row is a trial, len = 2000
  # each column is a time point, delta t = 1 ms
  x_copy = x.copy()
  
  # use standard error to balance effects of sample size in baseline
  x_score = np.zeros((x.shape[0],5))
  pinfo = []
  for i in range(x.shape[0]):
    if i % 100 == 0:
      toc = t() - tic
      print(f'{i}/{x.shape[0]} trials processed, time = {toc:.3f}')
    trial = x[i,:]
    x_score[i,:],p = score(trial,args.cutoff)
    pinfo.append(p)
  
  x_score = pd.DataFrame(x_score, columns = ['peak','height','auc','max_min','onset'])
  x_score.to_csv(args.out, sep = '\t', index = False, header = True)
  pinfo = pd.concat(pinfo, axis = 0).reset_index(drop = True)
  pinfo.to_csv(args.out.replace('.txt','_peakinfo.txt'), index = False, header = True)
  
  toc = t() - tic
  print(f'{x.shape[0]}/{x.shape[0]} trials processed, time = {toc:.3f}')
  
  # output diagnostics
  if True:
    smooth_CR = []
    for j in range(20,len(x_score.peak)):
      smooth_CR.append(sum(x_score.peak[j-20:j])/20)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    x_score['auc'] = x_score['auc']/x_score.auc.max()
    _,ax = plt.subplots(5,1,figsize = (12,30))
    sns.histplot(x_score[['peak','height','auc','max_min']], ax = ax[0],element = 'poly')
    ax[0].set_title(f'Diagnostic figure, cutoff = {args.cutoff:.4f}')
    sns.scatterplot(x_score[['peak','height','auc','max_min']], ax = ax[1])
    ax[1].set_xlabel('trials')
    ax[1].set_ylabel('score')
    ax[1].plot(np.arange(20,x.shape[0]),smooth_CR, '-k')
    sns.scatterplot(x_score, x = x_score.index, y = 'onset', ax = ax[2])
    ax[1].set_xlabel('trials')
    ax[1].set_ylabel('start_time')
    sns.scatterplot(pinfo,x = 'left_d2',y = 'peak_heights', ax = ax[3])
    # example traces
    for j in range(1,x.shape[0],int(x.shape[0]/5)):
      tmps = '+' if x_score.peak[j] > 0 else '-'
      lb = f'Trial {j+1}, peak {tmps}, height {x_score.height[j]:.3f}, start {x_score.onset[j]:.0f}'
      ax[4].plot(np.arange(-1500,450),x_copy[j,50:], label = lb)
    ax[4].axvspan(-550, 0, fc = '#ffffdf', color = '#ffffdf')
    ax[4].legend()
    
    plt.savefig(args.out.replace('txt','png'))
    
    toc = t() - tic
    print(f'Main diagnostic figure plotted, time = {toc:.3f}')
  
  if args.diag:
    try: os.mkdir('/home/yh464/rds/hpc-work/sae/diagnostics')
    except: pass
    os.chdir('/home/yh464/rds/hpc-work/sae/diagnostics')
    os.system('rm -rf *')
    # single-trial diagnostics
    rng = np.random.default_rng(2024)
    _,ax = plt.subplots(figsize=(12,24))
    for j in range(x.shape[0]):
      tmps = '+' if x_score.peak[j] > 0 else '-'
      lb = f'Trial {j+1}, peak {tmps}, height {x_score.height[j]:.3f}, start {x_score.onset[j]:.0f}'
      ax.plot(np.arange(-1550,450),x_copy[j,:], label = lb)
      if j % 5 == 0:
        ax.axvspan(-550, 0, fc = '#ffffdf', color = '#ffffdf')
        ax.legend()
        plt.savefig(f'/home/yh464/rds/hpc-work/sae/diagnostics/{rng.random():.5f}.png')
        plt.close()
        _,ax = plt.subplots(figsize=(12,24))
        if j % 50 == 0:
          toc = t() - tic
          print(f'{j} figures plotted after {toc:.3f} seconds')
  
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description = 'scores a single file')
  parser.add_argument('cutoff', type = float, help = 'peak cutoff')
  parser.add_argument('-i','--in', dest = '_in', 
                      help = 'input txt file, each row = 1 trial, each column = 1 ms')
  parser.add_argument('-o','--out',dest = 'out',
                      help = 'output file')
  parser.add_argument('-f','--force', action = 'store_true', default = False, 
                      help = 'force overwrite', dest = 'force')
  parser.add_argument('-d','--diagnostics', action = 'store_true', default = False,
                      help = 'diagnostics', dest = 'diag')
  args = parser.parse_args()
  
  # args._in = '../test/test.txt'
  # args.out = '../test/test_out.txt'
  # args.force = True
  main(args)