import numpy as np
import matplotlib.pyplot as plt
import json

# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        loss = []
        top_1_acc = []
        top_5_acc = []
        val_top_1_acc = []
        val_top_5_acc = []
        while True:
                line = f.readline()
                if line == '':
                    break
                b = json.loads(line)
                if b['mode'] == "train" :
                    top_1_acc.append(b['top1_acc'])
                    top_5_acc.append(b['top5_acc'])
                    loss.append(b['loss'])
                    # print(b['top1_acc'], b['top5_acc'],b['loss'])
                else:
                    val_top_1_acc.append(b['top1_acc'])
                    val_top_5_acc.append(b['top5_acc'])
    return loss, top_1_acc, top_5_acc, val_top_1_acc, val_top_5_acc



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

    face_loss, face_top_1_acc, face_top_5_acc, face_val_top_1_acc, face_val_top_5_acc = data_read(face_path)
    pose_loss, pose_top_1_acc, pose_top_5_acc, pose_val_top_1_acc, pose_val_top_5_acc = data_read(pose_path)
    # fp_loss, fp_top_1_acc, fp_top_5_acc, fp_val_top_1_acc, fp_val_top_5_acc = data_read(fp_path)

    # y_train_loss = data_read(train_loss_path)
    # y_train_acc = data_read(train_acc_path)

    face_x_train_loss = range(len(face_loss))
    face_x_train_acc = multiple_equal(face_x_train_loss, range(len(face_top_1_acc)))

    pose_x_train_loss = range(len(pose_loss))
    pose_x_train_acc = multiple_equal(pose_x_train_loss, range(len(pose_top_1_acc)))

    # fp_x_train_loss = range(len(fp_loss))
    # fp_x_train_acc = multiple_equal(fp_x_train_loss, range(len(fp_top_1_acc)))

    face_x_val_acc = range(len(face_val_top_1_acc))

    pose_x_val_acc = range(len(pose_val_top_1_acc))


    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')
    plt.ylabel('ACC')

    # plt.plot(pose_x_train_loss, pose_loss, linewidth=1, linestyle="solid", label="train pose_loss")
    plt.plot(face_x_val_acc, face_val_top_1_acc,  color='red', linestyle="solid", label="val face_top_1_acc")
    plt.plot(face_x_val_acc, face_val_top_5_acc,  color='blue', linestyle="solid", label="val face_top_5_acc")
    plt.legend()

    plt.title('Val Face ACC curve')

    save = './val_face_acc.png'

    plt.savefig(save)

    plt.show()
