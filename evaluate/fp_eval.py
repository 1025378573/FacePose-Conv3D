import numpy as np
import matplotlib.pyplot as plt
import json

# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        loss = []
        rgb_loss_cls = []
        pose_loss_cls = []
        rgb_top1_acc = []
        pose_top1_acc = []
        RGBPose_1_1_top1_acc = []
        RGBPose_2_1_top1_acc = []
        RGBPose_1_2_top1_acc = []
        while True:
                line = f.readline()
                if line == '':
                    break
                b = json.loads(line)
                if b['mode'] == "train" :
                    rgb_top1_acc.append(b['rgb_top1_acc'])
                    rgb_loss_cls.append(b['rgb_loss_cls'])
                    pose_top1_acc.append(b['pose_top1_acc'])
                    pose_loss_cls.append(b['pose_loss_cls'])
                    loss.append(b['loss'])
                    # print(b['top1_acc'], b['top5_acc'],b['loss'])
                else:
                    RGBPose_1_1_top1_acc.append(b['RGBPose_1:1_top1_acc'])
                    RGBPose_2_1_top1_acc.append(b['RGBPose_2:1_top1_acc'])
                    RGBPose_1_2_top1_acc.append(b['RGBPose_1:2_top1_acc'])
    return rgb_top1_acc, pose_top1_acc, rgb_loss_cls, pose_loss_cls, loss, RGBPose_1_1_top1_acc, RGBPose_2_1_top1_acc, RGBPose_1_2_top1_acc



# # 不同长度数据，统一为一个标准，倍乘x轴
def multiple_equal(x, y):
    x_len = len(x)
    y_len = len(y)
    times = x_len/y_len
    y_times = [i * times for i in y]
    return y_times


if __name__ == "__main__":

    face_path = '/home/chenyanting/pyskl/my_mod/evaluate/face.json'
    pose_path = '/home/chenyanting/pyskl/my_mod/evaluate/pose.json'
    fp_path = '/home/chenyanting/pyskl/my_mod/evaluate/face_pose.json'

    rgb_top1_acc, pose_top1_acc, rgb_loss_cls, pose_loss_cls, loss, RGBPose_1_1_top1_acc, RGBPose_2_1_top1_acc, RGBPose_1_2_top1_acc = data_read(fp_path)
    print(RGBPose_1_1_top1_acc)

    fp_x_train_loss = range(len(loss))
    fp_x_train_acc = multiple_equal(fp_x_train_loss, range(len(rgb_top1_acc)))

    fp_x_val_acc = range(len(RGBPose_1_1_top1_acc))



    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')
    plt.ylabel('Losee')

    plt.plot(fp_x_train_loss, rgb_loss_cls, linewidth=1, color='black', linestyle="solid", label="train rgb_loss_cls")
    plt.plot(fp_x_train_loss, pose_loss_cls, linewidth=1, color='blue', linestyle="solid", label="train pose_loss_cls")
    plt.plot(fp_x_train_loss, loss, linewidth=1, color='red', linestyle="solid", label="train loss")
    # plt.plot(fp_x_val_acc, RGBPose_1_1_top1_acc,  color='red', linestyle="solid", label="val RGBPose_1_1_top1_acc")
    # plt.plot(fp_x_val_acc, RGBPose_2_1_top1_acc,  color='blue', linestyle="solid", label="val RGBPose_2_1_top1_acc")
    # plt.plot(fp_x_val_acc, RGBPose_1_2_top1_acc,  color='yellow', linestyle="solid", label="val RGBPose_1_2_top1_acc")
    # plt.plot(fp_x_train_acc, rgb_top1_acc,  color='red', linestyle="solid", label="train rgb_top1_acc")
    # plt.plot(fp_x_train_acc, pose_top1_acc,  color='blue', linestyle="solid", label="train pose_top1_acc")
    plt.legend()

    plt.title('Train FP LOSS curve')

    save = './FP_loss.png'

    plt.savefig(save)

    plt.show()
