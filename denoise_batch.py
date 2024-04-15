# -*- coding: utf-8 -*-
#!/usr/bin/env python3

'''
denoise_batch.py
This script de-noises data in batches
'''

def main(args):
  import os
  from fnmatch import fnmatch
  
  scriptsdir = os.path.realpath('.')+ '/'
  if args.force: f = ' -f'
  else: f = ''
  
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
    if fnmatch(x, '*.txt'): flist.append(x)
  
  for x in flist:
    in_fname = os.path.realpath(x)
    out_fname = in_fname.replace(args._in, args.out).replace('.txt','_denoised.txt')
    if os.path.isfile(out_fname) and not args.force: continue
    os.system(
      # 'sbatch -N 1 -n 1 -c 1 -t 00:15:00 -p sapphire -o /dev/null/ '+
      'bash '+
      f'{scriptsdir}pymaster.sh '+
      f'{scriptsdir}denoise.py -i {in_fname} -o {out_fname} {f}')
    
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description = 'de-noises a single file')
  parser.add_argument('subj', help = 'subject ID')
  parser.add_argument('-i','--in', dest = '_in', default = '../extract',
                      help = 'input directory')
  parser.add_argument('-o','--out',dest = 'out', default = '../denoised',
                      help = 'output directory')
  parser.add_argument('-f','--force', action = 'store_true', default = False, help = 'force overwrite')
  args = parser.parse_args()
  main(args)