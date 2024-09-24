import argparse
import cv2
import mmcv
import numpy as np
import os
import os.path as osp
import shutil
import torch
import warnings
from scipy.optimize import linear_sum_assignment

from pyskl.apis import inference_recognizer, init_recognizer


vid_path = '/home/sharing/Datasets/MIntRec-video/rgb/S04_E01_31.mp4'

def frame_extraction(video_path, short_side):



    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))

        frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames

    a,b = frame_extraction(vid_path, 480)

def test(path):
    cap = cv2.VideoCapture(path)

    fps =int(cap.get(cv2.CAP_PROP_FPS))

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    videoWriter = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('X','V','I','D'), fps, size)

    ret, frame = cap.read()

    while(ret):

        # 展示一帧


        videoWriter.write(frame)

        cv2.waitKey(fps)

        ret,frame = cap.read()

    cap.release()

    cv2.destroyAllWindows()