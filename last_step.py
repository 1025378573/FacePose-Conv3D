from mmcv import load, dump
from pyskl.smp import *

os.chdir('/home/chenyanting/pyskl/my_mod/new_data')
train = load('train.json')
test = load('test.json')
dev = load('val.json')
# test_toy = load('test_toy.json')
annotations_train = load('/home/chenyanting/pyskl/new_mode/data/all.pkl')
# annotations_test = load('test.pkl')
# annotations_dev = load('val.pkl')
# annotations_test_toy = load('test_toy.pkl')
split = dict()
split['xsub_train'] = [x['vid_name'] for x in train]
split['xsub_test'] = [x['vid_name'] for x in test]
split['xsub_val'] = [x['vid_name'] for x in dev]
dump(dict(split=split, annotations=annotations_train), '/home/chenyanting/pyskl/new_mode/data/final.pkl')