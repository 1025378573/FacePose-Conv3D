from tqdm import tqdm
import pandas as pd
import os
from mmcv import load, dump
from pyskl.smp import *
import json
import shutil

ann_file_train = '/home/chenyanting/pyskl/my_mod/train_video.txt'
ann_file_val = '/home/chenyanting/pyskl/my_mod/val_video.txt'
ann_file_test = '/home/chenyanting/pyskl/my_mod/test_video.txt'

root = '/home/sharing/Datasets/MIntRec-video/rgb/'

def gen_txt(ann_file, root):

        with open(ann_file, 'r') as fin:
            for line in fin:
                
                line_split = line.strip().split()
                filename, label = line_split
                tmp = filename.split('/')[-1]
                new_name = '_'.join(filename.split('/')[-3:])

                old = os.path.join(root,tmp)
                new = os.path.join(root,new_name)

                shutil.copy(filename, root)
                os.rename(old, new)
                print(old,new)

    

# gen_txt(ann_file_train, root)
gen_txt(ann_file_val, root)
gen_txt(ann_file_test, root)