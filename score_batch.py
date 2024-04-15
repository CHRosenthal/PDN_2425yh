# -*- coding: utf-8 -*-
#!/usr/bin/env python3

'''
score_batch.py
This script scores data in batches
'''

def main(args):
  import os
  import numpy as np
  from fnmatch import fnmatch
  
  scriptsdir = os.path.realpath('.')+ '/'
  
  f = ' -f' if args.force else ''
  d = ' -d' if args.diag else ''
  
  # fail-safe design
  args._in = os.path.realpath(args._in) + '/'
  args.out = os.path.realpath(args.out) + '/'
  if not os.path.isdir(args.out): os.mkdir(args.out)
  os.chdir(args.out)
  if not os.path.isdir(args.subj): os.mkdir(args.subj)
  
  os.chdir(args._in)
  os.chdir(args.subj)
  
  flist = []
  for x in os.listdir():
    if fnmatch(x, '*_denoised.txt'): flist.append(x)
  
  # estimate cutoff
  tlist = []
  for x in flist:
    tlist.append(np.loadtxt(x))
  tlist = np.concatenate(tlist, axis = 0)
  trial_delta = tlist.max(axis=1) - tlist.min(axis = 1)
  trial_delta.sort()
  cutoff = trial_delta[-int(len(trial_delta)/4)] / 20
  
  for x in flist:
    in_fname = os.path.realpath(x)
    out_fname = in_fname.replace(args._in, args.out).replace('denoised','score')
    if os.path.isfile(out_fname) and not args.force: continue
    os.system(
      # 'sbatch -N 1 -n 1 -c 1 -t 00:15:00 -p sapphire -o /dev/null/ '+
      'bash '+
      f'{scriptsdir}pymaster.sh '+
      f'{scriptsdir}score.py {cutoff:.5f} -i {in_fname} -o {out_fname} {f}{d}')
    
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description = 'de-noises a single file')
  parser.add_argument('subj', help = 'subject ID')
  parser.add_argument('-i','--in', dest = '_in', default = '../denoised',
                      help = 'input directory')
  parser.add_argument('-o','--out',dest = 'out', default = '../score',
                      help = 'output directory')
  parser.add_argument('-d','--diagnostics', action = 'store_true', default = False,
                      help = 'diagnostics', dest = 'diag')
  parser.add_argument('-f','--force', action = 'store_true', default = False, help = 'force overwrite')
  args = parser.parse_args()
  main(args)