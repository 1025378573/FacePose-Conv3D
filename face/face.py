
# 导入相关库
import cv2
import numpy as np
from model.faceDetector.s3fd import S3FD
import warnings
warnings.filterwarnings("ignore")

# 读取视频
video = cv2.VideoCapture('/home/sharing/Datasets/MIntRec-video/rgb/S04_E01_205.mp4')

# fps =int(video.get(cv2.CAP_PROP_FPS))
fps = 20
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
face_Size = (240,240)

# 加载人脸检测器
# face_cascade = cv2.CascadeClassifier('/home/chenyanting/anaconda3/envs/open-mmlab/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')
DET = S3FD(device='cuda')

# 创建 VideoWriter 对象，用于写入视频
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter('output2.avi', fourcc, fps, face_Size)

count  = 0
path = '/home/sharing/Datasets/MIntRec-video/frames/mmaction2/frames/S04_E01_137/img_000049.jpg'

(xx1,yy1,xx2,yy2) = (857,193,1310,845)

frame = cv2.imread(path)
# imageNumpy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

person = frame[yy1:yy2,xx1:xx2]
imageNumpy = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
bboxes = DET.detect_faces(imageNumpy, conf_th=0.8, scales=[0.25])

cv2.imwrite("./data3/."+str(111)+'.jpg',imageNumpy)
cv2.imwrite("./data3/."+str(112)+'.jpg',person)
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)[0]
print(bboxes)
x = int(bboxes[0][0])
y = int(bboxes[0][1])
w = int(bboxes[0][2])
h = int(bboxes[0][3])
    # cv2.rectangle(test, (x,y), (x+w, y+h), (255,0,0), 0)

    # cv2.imwrite("./data3/."+str(9)+'.jpg',test)
face_test = person[y:y+h,x:x+h]
new_face = cv2.resize(person[y:y+h,x:x+w],face_Size)
# face = test[y:y+h,x:x+w]
print(new_face.shape)
cv2.imwrite("./data4/."+str(111)+'.jpg',new_face)
cv2.imwrite("./data4/."+str(112)+'.jpg',face_test)

# while True:
#     # 读取一帧
#     ret, frame = video.read()
#     if frame is None:
#         break
    
#     # 人脸检测
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     count = count+1
#     # 在检测到的人脸区域绘制矩形
    
#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 0)

#         cv2.imwrite("./data3/."+str(count)+'.jpg',frame)
    
#         face = cv2.resize(frame[y:y+h,x:x+w],face_Size)
#         cv2.imwrite("./data/."+str(count)+'.jpg',face) 
#         print(frame.shape, face.shape, count)
#     # 将检测到的人脸区域写入视频
#         out.write(face)

# # 释放视频
# video.release()
# out.release()
 
