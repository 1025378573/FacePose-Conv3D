from tqdm import tqdm
import pandas as pd
import os
from mmcv import load, dump
from pyskl.smp import *
import json

path1 = '/home/chenyanting/pyskl/my_mod/train_video.txt'
path2 = '/home/chenyanting/pyskl/my_mod/val_video.txt'
path3 = '/home/chenyanting/pyskl/my_mod/test_video.txt'

root = '/home/sharing/Datasets/MIntRec-video/face_video/'

def gen_txt(ann_file, root, name):
        lis = []
        with open(ann_file, 'r') as fin:
            for line in fin:
                dic = {}
                line_split = line.strip().split()
                filename, label = line_split
                tmp = filename.split('/')[-1]
                new_name = '_'.join(filename.split('/')[-3:])
                new_name = new_name.replace('.mp4', '.avi')
                dic['path'] = new_name
                dic['label'] = label
                lis.append(dic)

        tmpl =os.path.join(root,'{}')
        lines = [(tmpl + ' {}').format(x['path'], x['label']) for x in lis] 
        mwlines(lines, ('{}.list').format(name))
            
name1 = 'train'
name2 = 'val'
name3 = 'test'
gen_txt(path1, root, name1)
gen_txt(path2, root, name2)
gen_txt(path3, root, name3)