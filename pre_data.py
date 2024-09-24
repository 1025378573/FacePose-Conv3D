from tqdm import tqdm
import pandas as pd
import os
from mmcv import load, dump
from pyskl.smp import *
import json

train_file_name = '/home/sharing/disk3/zhanghanlei/Datasets/MIntRec/repos/mmaction2/train.csv'
dev_file_name = '/home/sharing/disk3/zhanghanlei/Datasets/MIntRec/repos/mmaction2/dev.csv'
test_file_name = '/home/sharing/disk3/zhanghanlei/Datasets/MIntRec/repos/mmaction2/test.csv'
test_toy_name = '/home/sharing/disk3/zhanghanlei/Datasets/MIntRec/repos/mmaction2/test_toy.csv'

vid_root = '/home/sharing/disk3/zhanghanlei/Datasets/MIntRec/private/raw_video_data'

def load_csv(file_name):
    df = pd.read_csv(file_name, header=None)
    return df


def gen_list(df, vid_root, name):
        tmp = []
        name = str(name)
        for i in tqdm(range(len(df)), desc = 'file'):
            dic = {}

            if i == 0:
                vid_name = df[0][0]
                vid_path = df[0][i]+'.mp4'
                sp = vid_path.split('_')
                path = '/'.join(sp)
                # gt_path = os.path.join(vid_root, path)

                lable = df[6][i]

                dic['path'] = path
                dic['label'] = lable
                tmp.append(dic)               

            if vid_name != df[0][i]:
                vid_name = df[0][i]
                vid_path = df[0][i]+'.mp4'
                sp = vid_path.split('_')
                path = '/'.join(sp)
                # gt_path = os.path.join(vid_root, path)

                lable = df[6][i]

                dic['path'] = path
                dic['label'] = lable
                tmp.append(dic)

            if i == len(df)-1:
                break

        tmpl =os.path.join(vid_root,'{}')
        lines = [(tmpl + ' {}').format(x['path'], x['label']) for x in tmp] 
        mwlines(lines, ('{}.list').format(name))


                
def gen_json(df, jsonpath):
        tmp = []
        for i in tqdm(range(len(df)), desc = 'file'):
            dic = {}

            if i == 0:
                vid_name = df[0][0]
                lable = df[6][i]
                start_frame = df[8][i]
                end_frame = df[9][i]

                dic['vid_name'] = vid_name
                dic['label'] = int(lable)
                dic['start_frame'] = int(start_frame)
                dic['end_frame'] = int(end_frame)
                tmp.append(dic)               

            if vid_name != df[0][i]:
                vid_name = df[0][i]
                lable = df[6][i]
                start_frame = df[8][i]
                end_frame = df[9][i]

                dic['vid_name'] = vid_name
                dic['label'] = int(lable)
                dic['start_frame'] = int(start_frame)
                dic['end_frame'] = int(end_frame)

                tmp.append(dic)

            if i == len(df)-1:
                break
        
        with open(jsonpath, 'w') as outfile:
            json.dump(tmp, outfile)



# df_train = load_csv(train_file_name)
# df_test = load_csv(test_file_name)
# df_dev = load_csv(dev_file_name)
df_test_toy = load_csv(test_toy_name)

# train_jsonpath = 'train.json'
# test_jsonpath = 'test.json'
# dev_jsonpath = 'dev.json'
test_toy_jsonpath = 'test_toy.json'

# gen_list(df_train, vid_root, 'train')
# gen_list(df_test, vid_root, 'test')
# gen_list(df_dev, vid_root, 'dev')
gen_list(df_test_toy, vid_root, 'test_toy')

# gen_json(df_train, train_jsonpath)
# gen_json(df_test, test_jsonpath)
# gen_json(df_dev, dev_jsonpath)
gen_json(df_test_toy, test_toy_jsonpath)