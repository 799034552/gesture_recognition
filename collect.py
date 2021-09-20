import cv2
import numpy as np


# 肤色检测函数
def skinFind(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) # 把图像转换到YCrCb色域
    skin3 = cv2.inRange(ycrcb,(0,140,77),(255,173,130))
    return skin3

camera = cv2.VideoCapture(0)
camera.set(10,200)
n = 20
while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.bilateralFilter(frame, 5, 50, 100) 
    temp = skinFind(frame)
    cX =int(0.45 * frame.shape[1])
    cY =int(0.2 * frame.shape[0])
    cv2.rectangle(temp, (cX, cY),(cX + 300, cY + 300), (255, 0, 0), 2)
    cv2.imshow('out',temp)
    frame = cv2.bilateralFilter(frame, 5, 50, 100) 
    cv2.rectangle(frame, (cX, cY),(cX + 300, cY + 300), (255, 0, 0), 2)
    cv2.imshow('original', frame)
    img = temp[cY:cY + 300,cX:cX + 300]
    img = cv2.medianBlur(img, 5)  # 中值滤波
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('img', img)
    k = cv2.waitKey(50)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('s'):
        n += 1
        cv2.imwrite("./pics/1_"+str(n)+".jpg", img)