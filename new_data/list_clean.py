
# 定义 txt 文件和 list 文件路径
txt_file = '/home/chenyanting/pyskl/my_mod/new_data/inval3.txt'
list_file = '/home/chenyanting/pyskl/my_mod/new_data/test.list'
root = '/home/sharing/Datasets/MIntRec-video/face_video/'
# 读取 txt 文件中要删除的视频名称
with open(txt_file, 'r') as f:
    video_names = {root + line.strip()+ '.avi' for line in f}

# 根据 txt 文件中的视频名称，读取 list 文件，删除匹配的行
with open(list_file, 'r') as f:
    lines = f.readlines()

with open(list_file, 'w') as f:
    for line in lines:
        line_split = line.strip().split()
        filename, label = line_split
        if not any(video_name in filename for video_name in video_names):
            f.write(line)
            print(f'Kept video: {line.strip()}')
        else:
            print(f'Deleted video: {line.strip()}')
