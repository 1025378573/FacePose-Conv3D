import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from mmcv import load, dump
from pyskl.smp import *
import json
from model.faceDetector.s3fd import S3FD

path1 = '/home/chenyanting/pyskl/data/mmaction2/dev.csv'
path2 = '/home/chenyanting/pyskl/data/mmaction2/train.csv'
path3 = '/home/chenyanting/pyskl/data/mmaction2/test.csv'
root = '/home/sharing/Datasets/MIntRec-video/frames/mmaction2/frames'
new_path = '/home/sharing/Datasets/MIntRec-video/test'
face_video = '/home/sharing/Datasets/MIntRec-video/face_video'
# face_cascade = cv2.CascadeClassifier('/home/chenyanting/anaconda3/envs/open-mmlab/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_det(path, roi):
    
    face_Size = (240,240)

    # face_cascade = face_cascade
    DET = S3FD(device='cuda')

    frame = cv2.imread(path)

    (xx1,yy1,xx2,yy2) = roi
    person = frame[yy1:yy2,xx1:xx2]
    imageNumpy = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./data5/."+str(112)+'.jpg',person)
    # gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bboxes = DET.detect_faces(imageNumpy, conf_th=0.8, scales=[0.25])
    # for (x,y,w,h) in faces: 
    #     face = cv2.resize(person_roi[y:y+h,x:x+w],face_Size)
    x = int(bboxes[0][0])
    y = int(bboxes[0][1])
    w = int(bboxes[0][2])
    h = int(bboxes[0][3])

    face = cv2.resize(person[y:y+h,x:x+h],face_Size)
    cv2.imwrite("./data5/."+str(113)+'.jpg',person[y:y+h,x:x+h])
    print(face.shape)
    return face, frame

def gen_dic(file_name):
        df = pd.read_csv(file_name, header=None)
        dic_fin = {}
        for i in tqdm(range(len(df)), desc = 'file'):
            dic = {}              
            

            if i == len(df)-1:
                vid_name = df[0][i]
                image_id = int(df[1][i])
                x1 = int(df[2][i])
                y1 = int(df[3][i])
                x2 = int(df[4][i])
                y2 = int(df[5][i])
                s="%06d" % image_id
                image_name='img_'+s+'.jpg'

                dic[image_name] = (x1,y1,x2,y2)
                frame.append(dic)
                dic_fin[vid_name] = frame
                break

            if i == 0:
                frame = []
                vid_name = df[0][i]
                image_id = int(df[1][i])
                x1 = int(df[2][i])
                y1 = int(df[3][i])
                x2 = int(df[4][i])
                y2 = int(df[5][i])
                s="%06d" % image_id
                image_name='img_'+s+'.jpg'

                dic[image_name] = (x1,y1,x2,y2)

                frame.append(dic)
                

            elif vid_name != df[0][i]:
                dic_fin[vid_name] = frame
 
                dic = {}
                frame = []
                vid_name = df[0][i]
                image_id = int(df[1][i])
                x1 = int(df[2][i])
                y1 = int(df[3][i])
                x2 = int(df[4][i])
                y2 = int(df[5][i])
                s="%06d" % image_id
                image_name='img_'+s+'.jpg'

                dic[image_name] = (x1,y1,x2,y2)
                frame.append(dic)


            elif vid_name == df[0][i]:
                vid_name = df[0][i]
                image_id = int(df[1][i])
                x1 = int(df[2][i])
                y1 = int(df[3][i])
                x2 = int(df[4][i])
                y2 = int(df[5][i])
                s="%06d" % image_id
                image_name='img_'+s+'.jpg'

                dic[image_name] = (x1,y1,x2,y2)

                frame.append(dic)


        # print(len(dic_fin))
        # print(dic_fin['S05_E14_14'])
        return dic_fin


dev = gen_dic(path1)
train = gen_dic(path2)
test = gen_dic(path3)
lis = []
# 创建 VideoWriter 对象，用于写入视频
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

fps = 20
Size = (240,240)
# print(len(train), len(test))
# print('sdsd')
for key in tqdm(dev.keys(), desc = 'dev'):
    print(key)
    vid_name = new_path+'/'+key+'.avi'
    out = cv2.VideoWriter(vid_name, fourcc, fps, Size)
    for dic in dev[key]:
        for k, v in dic.items():
            pp = os.path.join(key,k)
            img_path = os.path.join(root,pp)
            print(img_path)
            # face, frame = face_det(img_path, v)
            
            # cv2.imwrite("./data5/"+k+'.jpg',face)
            # out.write(face)
            try:
                face, frame = face_det(img_path, v)
                
                # cv2.imwrite("./data5/"+k+'.jpg',face)
                out.write(face)
            except:
                lis.append(pp)
                continue
    out.release()

for key in tqdm(train.keys(), desc = 'train'):
    vid_name = new_path +'/'+key+'.avi'
    out = cv2.VideoWriter(vid_name, fourcc, fps, Size)
    for dic in train[key]:
        for k, v in dic.items():
            pp = os.path.join(key,k)
            img_path = os.path.join(root,pp)
            try:
                face, frame = face_det(img_path, v)
                
                # cv2.imwrite("./data4/"+k+'.jpg',face)
                out.write(face)
            except:
                lis.append(pp)
                continue
    out.release()

for key in tqdm(test.keys(), desc = 'test'):
    vid_name = new_path +'/'+key+'.avi'
    out = cv2.VideoWriter(vid_name, fourcc, fps, Size)
    for dic in test[key]:
        for k, v in dic.items():
            pp = os.path.join(key,k)
            img_path = os.path.join(root,pp)
            try:
                face, frame = face_det(img_path, v)
                
                # cv2.imwrite("./data4/"+k+'.jpg',face)
                out.write(face)
            except:
                lis.append(pp)
                continue
    out.release()

f=open("/home/chenyanting/pyskl/my_mod/face/invalid.txt","w")
for line in lis:
    f.write(line+'\n')
f.close()