

import os

# 定义 txt 文件路径和视频文件夹路径
txt_file = '/home/chenyanting/pyskl/my_mod/new_data/inval3.txt'
video_folder1 = '/home/sharing/Datasets/MIntRec-video/face_video/'
video_folder2 = '/home/sharing/Datasets/MIntRec-video/face_only/'

# 读取 txt 文件
with open(txt_file, 'r') as f:
    video_names = [line.strip() for line in f.readlines()]


# 遍历视频文件夹，查找并删除 txt 文件中出现的视频
for dirpath, dirnames, filenames in os.walk(video_folder1):
    # print(filenames)
    for filename in filenames:
        name = filename.split('.')[0]
        if name in video_names:
            file_path = os.path.join(dirpath, filename)
            # print(file_path)
            os.remove(file_path)
            print(f'Deleted video: {file_path}')

