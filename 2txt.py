from tqdm import tqdm
import pandas as pd
import os
from mmcv import load, dump
from pyskl.smp import *
import json

ann_file_train = '/home/chenyanting/pyskl/my_mod/data/train.list'
ann_file_val = '/home/chenyanting/pyskl/my_mod/data/dev.list'
ann_file_test = '/home/chenyanting/pyskl/my_mod/data/test.list'
vid_root = '/home/sharing/Datasets/MIntRec-video/raw_video_data'

def gen_txt(ann_file, vid_root, name):
        tmp = []
        with open(ann_file, 'r') as fin:
            for line in fin:
                dic = {}
                line_split = line.strip().split()
                filename, label = line_split
                label = int(label)
                path = '/'.join(filename.split('/')[-3:])


                dic['path'] = path
                dic['label'] = label
                tmp.append(dic)     

        tmpl =os.path.join(vid_root,'{}')
        lines = [(tmpl + ' {}').format(x['path'], x['label']) for x in tmp] 
        mwlines(lines, ('{}.txt').format(name))

# def load_annotations(ann_file):

#     video_infos = []
#     with open(ann_file, 'r') as fin:
#         for line in fin:
#             line_split = line.strip().split()
#             filename, label = line_split
#             label = int(label)
#             path = '/'.join(filename.split('/')[-3:])

#             tmpl =os.path.join(vid_root,'{}')
#             print(path, label)
    

gen_txt(ann_file_train, vid_root, 'train_video')
gen_txt(ann_file_val, vid_root, 'val_video')
gen_txt(ann_file_test, vid_root, 'test_video')