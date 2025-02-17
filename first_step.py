# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
# import pdb
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_top_down_pose_model, init_pose_model
import decord
import mmcv
import numpy as np
# import torch.distributed as dist
from tqdm import tqdm
# import mmdet
# import mmpose
# from pyskl.smp import mrlines
import cv2
 
from pyskl.smp import mrlines
 
 
def extract_frame(video_path):
    vid = decord.VideoReader(video_path)
    return [x.asnumpy() for x in vid]
 
 
def detection_inference(model, frames):
    model=model.cuda()
    results = []
    for frame in frames:
        result = inference_detector(model, frame)
        results.append(result)
    return results
 
 
def pose_inference(model, frames, det_results):
    model=model.cuda()
    assert len(frames) == len(det_results)
    total_frames = len(frames)
    num_person = max([len(x) for x in det_results])
    kp = np.zeros((num_person, total_frames, 17, 3), dtype=np.float32)
 
    for i, (f, d) in enumerate(zip(frames, det_results)):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        for j, item in enumerate(pose):
            kp[j, i] = item['keypoints']
    return kp
 
 
def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
 
    parser.add_argument(
        '--det-config',
        default='faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
 
    parser.add_argument(
        '--det-ckpt',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
 
    parser.add_argument('--pose-config', type=str, default='hrnet_w32_coco_256x192.py')
    parser.add_argument('--pose-ckpt', type=str, default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'))

    parser.add_argument('--det-score-thr', type=float, default=0.7)

    parser.add_argument('--det-area-thr', type=float, default=1300)

    parser.add_argument('--video-list', type=str,default='/home/chenyanting/pyskl/my_mod/test_video.txt', help='the list of source videos')
    # * out should ends with '.pkl'
    parser.add_argument('--out', type=str, default='/home/chenyanting/pyskl/my_mod/use_talknet_data/all.pkl', help='output pickle name')
    parser.add_argument('--tmpdir', type=str, default='tmp')
    parser.add_argument('--local_rank', type=int, default=1)
    # pdb.set_trace()
 
    # if 'RANK' not in os.environ:
    #     os.environ['RANK'] = str(args.local_rank)
    #     os.environ['WORLD_SIZE'] = str(1)
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'
 
    args = parser.parse_args()
    return args
 
 
def main():
    args = parse_args()
    assert args.out.endswith('.pkl')
 
    lines = mrlines(args.video_list)
    lines = [x.split() for x in lines]
 
    assert len(lines[0]) in [1, 2]
    if len(lines[0]) == 1:
        annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0]) for x in lines]
    else:
        annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0], label=int(x[1])) for x in lines]
    # if len(lines[0]) == 1:
    #     annos = [dict(frame_dir=('_'.join(x[0].split('/')[9:]).split('.')[0]), filename=x[0]) for x in lines]
    # else:
    #     annos = [dict(frame_dir=('_'.join(x[0].split('/')[9:]).split('.')[0]), filename=x[0], label=int(x[1])) for x in lines]
 
    rank = 0  # 添加该
    world_size = 1  # 添加
 
    # init_dist('pytorch', backend='nccl')
    # rank, world_size = get_dist_info()
    #
    # if rank == 0:
    #     os.makedirs(args.tmpdir, exist_ok=True)
    # dist.barrier()
    my_part = annos
    # my_part = annos[rank::world_size]
    print("from det_model")
    det_model = init_detector(args.det_config, args.det_ckpt, 'cuda')
    assert det_model.CLASSES[0] == 'person', 'A detector trained on COCO is required'
    print("from pose_model")
    pose_model = init_pose_model(args.pose_config, args.pose_ckpt, 'cuda')
    n = 0
    for anno in tqdm(my_part):
        frames = extract_frame(anno['filename'])
        print("anno['filename", anno['filename'])
        det_results = detection_inference(det_model, frames)
        # * Get detection results for human
        det_results = [x[0] for x in det_results]
        for i, res in enumerate(det_results):
            # * filter boxes with small scores
            res = res[res[:, 4] >= args.det_score_thr]
            # * filter boxes with small areas
            box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
            assert np.all(box_areas >= 0)
            res = res[box_areas >= args.det_area_thr]
            det_results[i] = res
 
        pose_results = pose_inference(pose_model, frames, det_results)
        shape = frames[0].shape[:2]
        anno['img_shape'] = anno['original_shape'] = shape
        anno['total_frames'] = len(frames)
        anno['num_person_raw'] = pose_results.shape[0]
        anno['keypoint'] = pose_results[..., :2].astype(np.float16)
        anno['keypoint_score'] = pose_results[..., 2].astype(np.float16)
        anno.pop('filename')
 
    mmcv.dump(my_part, osp.join(args.tmpdir, f'part_{rank}.pkl'))
    # dist.barrier()
 
    if rank == 0:
        parts = [mmcv.load(osp.join(args.tmpdir, f'part_{i}.pkl')) for i in range(world_size)]
        rem = len(annos) % world_size
        if rem:
            for i in range(rem, world_size):
                parts[i].append(None)
 
        ordered_results = []
        for res in zip(*parts):
            ordered_results.extend(list(res))
        ordered_results = ordered_results[:len(annos)]
        mmcv.dump(ordered_results, args.out)
 
 
if __name__ == '__main__':
    # default_mmdet_root = osp.dirname(mmcv.__path__[0])
    # default_mmpose_root = osp.dirname(mmcv.__path__[0])
    main()
 