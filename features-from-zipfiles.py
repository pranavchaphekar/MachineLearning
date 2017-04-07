# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 05:43:00 2016

@author: elliott
"""

import os
from collections import Counter
from glob import glob
from zipfile import ZipFile

os.chdir('cleaned')
zipfiles = glob('*zip')

for zfname in zipfiles:        
    print(zfname)
    zfile = ZipFile(zfname)    
    year = zfname.split('/')[-1][:-4]
 
    members = zfile.namelist()        
    threshold = len(members) / 200    
    docfreqs = Counter()        
    for fname in members:
        
        # "maj" means this is the majority opinion
        if not fname.endswith('-maj.txt'):
            continue

        docid = fname.split('/')[-1][:-4]                     
        text = zfile.open(fname).read().decode()                
        
        # featurize
        
