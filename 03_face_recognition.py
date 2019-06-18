__author__ = "Luke Liu"
#encoding="utf-8"
import cv2
import numpy as np
import os
import  dlib

#  七牛云库的导入
from qiniu import Auth, put_file, etag, urlsafe_base64_encode
import qiniu.config
import json

#dlib 64个特征点需要的东西
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

send_mode="Can not catch your children"

path_xml=r'C:\ProgramData\Anaconda3\Lib\site-packages\cv2\data'
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(os.path.join(path_xml,cascadePath))
eyeCascade = cv2.CascadeClassifier(os.path.join(path_xml,"haarcascade_eye.xml"))
font = cv2.FONT_HERSHEY_SIMPLEX
# iniciate id counter
id = 0
zoushen=0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Liu Zihua','Ma Ruoyan','Liu Zihua(off Glasses)',"Lai Yueming","Feng Pengfhui",
         "Li Wenjie"]
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height
# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
whether_enter=False   #是否识别到了人脸
import  time
# 工程里面用的时间判断标志
a=time.time()  # this time is used for  人脸识别的
a_c=a          # this time is used for  闭眼检测的
begin=a       # this time is used for 每0.2s向 angel 里面加一个数据的
begin_c=a      # this time is used for 每4s 就做一次是否跑神的判断的
begin_S=a     # this time is used for 每4s Left_Right 为 True时候就给总的cnt加1用的
renew_cnt_time=a  # this time is used for reset the cnt's value every Per minute
angel=[]
Left_Right=False   # 是否现在正在东张西望。

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Flip vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    if len(faces)==0:
        whether_enter=False


#识别特征带点
    rects = detector(gray, 0)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])

        # 后期计算数字
        m1 = landmarks[0]
        m2= landmarks[28]
        dis_a_b = (m2- m1).getA().tolist()[0][0]
        print(dis_a_b)
        now=time.time()
        if now - begin > 0.2:
            angel.append(dis_a_b)
            angel = sorted(angel)
            begin = now

        now_c = time.time()

        if now_c - begin_c > 4:
            a1 = angel[0]
            b1 = angel[-1]

            if a1 < 100:
                if b1 > 200:
                    Left_Right = True
                    angel = []
                    begin_c = now_c

            if b1 - a1 < 100:
                Left_Right = False
                angel = []
                begin_c = now_c


        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])
            # 利用cv2.circle给每个特征点画一个圈，共68个
            cv2.circle(img, pos, 5, color=(0, 255, 0))
            # 利用cv2.putText输出1-68
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(idx), pos, font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)



    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=10,
            minSize=(5, 5),
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)

        if len(eyes)==0:
            c=time.time()
            # 闭上眼睛的时间超过了3s, 走神+1
            if c-a_c>3:
                zoushen+=1
                a_c=c
        else:
            a_c=time.time()
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 60):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            whether_enter=True

        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            whether_enter=False

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    if whether_enter == True:
        send_mode="Your children is sitting for study!"
        a=time.time()
        cv2.putText(img, "Your children is sitting for study!", (80, 70), font, 1, (0, 0, 255), 2)
    if whether_enter == False:

        # 超过5s没有识别到对象，走神+1
        send_mode="Sorry,Can not catch your children"
        b=time.time()
        if b-a>5:
            zoushen+=1
            a=b
        cv2.putText(img, "Can not catch your children ", (120, 70), font, 1, (0, 0, 255), 2)

        #每次都1次东张西望的举动，走神+1
    if Left_Right==True:
        bb=time.time()
        if bb-begin_S>4:
            zoushen+=1
            begin_S=bb

    cv2.putText(img,"Absent_minded NO {} per min".format(zoushen),(150,435),font,1,(255,0,0),2)
    cv2.imshow('camera', img)

    import json
    #定义要传入的字典的值
    dicts={"Children":send_mode,"Absent_minded times(per min)":zoushen}

    #每10s 读写一次json文件
    write_json_time=time.time()
    if write_json_time-renew_cnt_time>10:
        with open("chidren_condition.json",'w') as f1:
            f1.write(json.dumps(dicts))
        renew_cnt_time=write_json_time
        zoushen=0

        # upload to seven crow Cloud
        # 数值与ID
        access_key = 'RcNC14Omfl9-aWFzEZmKYS0Iz6QW_zJDTkgoiQ3L'
        secret_key = 'naVNF52jMXVm2Gvq05R9GNSQ-zEioMIG_CkM7G0r'
        q = Auth(access_key, secret_key)
        # 要上传的空间
        bucket_name = 'childrendtc'
        # 上传到七牛后保存的文件名
        key = 'chidren_condition.json'
        # 生成上传 Token，可以指定过期时间等
        token = q.upload_token(bucket_name, key, 3600)

        # 要上传文件的本地路径
        localfile = 'chidren_condition.json'
        rett, info = put_file(token, key, localfile)
        print(info)

    print("Whether enter: ",whether_enter)
    print("Whether Left_and_Right",Left_Right)
    print("absent_minded number is :",zoushen)
#-------------------------------------------------------------------------------------------------------------

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()