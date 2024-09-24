from tqdm import tqdm
import pandas as pd
import os
from mmcv import load, dump
from pyskl.smp import *
import json

path1 = '/home/chenyanting/pyskl/my_mod/new_data/train.list'
path2 = '/home/chenyanting/pyskl/my_mod/new_data/val.list'
path3 = '/home/chenyanting/pyskl/my_mod/new_data/test.list'

j1 = '/home/chenyanting/pyskl/my_mod/new_data/train.json'
j2 = '/home/chenyanting/pyskl/my_mod/new_data/val.json'
j3 = '/home/chenyanting/pyskl/my_mod/new_data/test.json'

def writeJson(ann_file,jsonpath):
    outpot_list=[]

    with open(ann_file, 'r') as fin:
        for line in fin:
            traindit = {}
            line_split = line.strip().split()
            filename, label = line_split
            traindit = {}
            tmp = filename.split('/')[-1]
            traindit['vid_name'] = tmp.replace('.avi', '')
            traindit['label'] = int(label)
            traindit['start_frame'] = 0
            
            vid = decord.VideoReader(filename)
            traindit['end_frame'] = len(vid)
            outpot_list.append(traindit.copy())
    print(len(outpot_list))
    with open(jsonpath, 'w') as outfile:
        json.dump(outpot_list, outfile)


writeJson(path1, j1)
writeJson(path2, j2)
writeJson(path3, j3)
