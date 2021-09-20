import os
import cv2
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
import math

# 寻找一组数据的峰值
def peakSearch(arr):
    peakThe = int(len(arr) * 0.032)
    peakList = []
    peakVal = []
    for i in range(len(arr)):
        error = 0
        if (arr[i] < 50):
            continue
        for k in range(1,peakThe):
            if i + k >= len(arr):
                t = arr[i + k - len(arr)]
            else:
                t = arr[i + k]
            if (arr[i - k] < 30 or t < 30): # 35
                error = 1
                break
            if (arr[i] < arr[i - k] or arr[i] < t):
                error = 1
                break
        if error == 0:
            if (len(peakList) > 0) and i - peakList[-1] < 30:
                continue
            peakVal.append(arr[i])
            peakList.append(i)
    return peakList

# 计算两个矢量的夹角
def findAngle(arr1, arr2):
    a = arr1[0] * arr2[0] + arr1[1] * arr2[1]
    b1 = math.sqrt(arr1[0] * arr1[0] + arr1[1] * arr1[1])
    b2 = math.sqrt(arr2[0] * arr2[0] + arr2[1] * arr2[1])
    return math.acos(a/b1/b2) * 180 / 3.14

def handleOnePic(picName):
    #读取图片
    frame = cv2.imread(picName)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    #滤波
    frame = cv2.blur(frame,(3,3))
    frame = cv2.medianBlur(frame, 5)
    #寻找最大轮廓，防止图像中有其他小物件的干扰
    contours, hierarchy = cv2.findContours(frame , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE )
    maxCount = []
    for item in contours:
        if len(item) > len(maxCount):
            maxCount = item
    maxCount = np.array(maxCount)
    area = cv2.contourArea(maxCount) / frame.size
    #分离x与y
    xs = maxCount[:,:,0]
    ys = maxCount[:,:,1]
    # 找出数据的质心
    mu=cv2.moments(maxCount)
    mc=[mu['m10'] / mu['m00'], mu['m01'] / mu['m00']]
    #对图像进行膨胀，让图像更“胖”
    kernel = np.ones((20, 20), np.uint8)
    frame = cv2.dilate(frame, kernel)
    # 寻找新图像的质心
    contours, hierarchy = cv2.findContours(frame , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE )
    maxCount = []
    for item in contours:
        if len(item) > len(maxCount):
            maxCount = item
    maxCount = np.array(maxCount)
    mu=cv2.moments(maxCount)
    mc2=[mu['m10'] / mu['m00'], mu['m01'] / mu['m00']]

    # 计算手掌中心的坐标
    x3 =mc[0] + 5*(mc[0] - mc2[0])
    k = (mc[1] - mc2[1]) / (mc[0] - mc2[0])
    y3 = k * (x3 - mc2[0]) + mc2[1]

    # 将数据变为整数
    x3 = int(x3)
    y3 = int(y3)
    cx = int(mc[0])
    cy = int(mc[1])

    # 求出所有数据对掌心的距离，用最大值进行对数据归一化
    distance = np.sqrt(np.multiply(xs - x3,xs - x3) + np.multiply(ys - y3,ys - y3))
    maxDistance = np.max(distance)
    minDistance = np.min(distance)
    tDistance = distance / maxDistance
    distance = (distance - minDistance) / (maxDistance - minDistance) * 100

    # 由于数据是无起始顺序的，对数据进行处理，连续低于某个值位于前方
    threshold_bottom = 50
    rightTimes = 30
    for i in range(0,len(distance)):
        if distance[i] > threshold_bottom:
            rightTimes = 20
        else:
            rightTimes -= 1
        if rightTimes == 1:
            t = distance[i:]
            distance = np.vstack((distance[i:],distance[0:i]))
            ys = np.vstack((ys[i:],ys[0:i]))
            xs = np.vstack((xs[i:],xs[0:i]))
            break

    # 寻找峰值
    peakList = peakSearch(distance)
    res = [] #返回的数据 分别为： 峰值个数 到掌心距离的平均值   峰值与手掌方向夹角的总和 峰值到掌心距离总和 掌心x 掌心y 手掌方向 图像占整个图像面积的比
    
    res.append(len(peakList))
    res.append(np.mean(tDistance))
    angles = 0 #计算所有峰值与手掌方向向量的夹角和
    stander =  (mc2[0] - mc[0], mc2[1] - mc[1]) # 手掌方向的矢量
    totalDis = 0 # 所有掌心到峰值的矢量长度和
    peakLists = [] #所有掌心到峰值的矢量
    for i in peakList:
        peakLists.append(float(xs[i]) - x3)
        peakLists.append(float(ys[i]) - y3)



    for i in range(int(len(peakLists)/2)):
        angle = findAngle(stander,(int(peakLists[2*i]), int(peakLists[2*i + 1])))
        if (angle > 92):
            res[0] -= 1
            continue
        angles += angle
        dis = math.sqrt((peakLists[2*i]) * (peakLists[2*i]) + (peakLists[2*i + 1]) * (peakLists[2*i + 1]))
        totalDis += dis
    
    res.append(angles)
    res.append(totalDis)
    res.append(x3)
    res.append(y3)
    res.append(- stander[1] / stander[0])
    res.append(area)
    return res

if __name__ == '__main__':
    allList = os.listdir('./pics')
    data = []
    for item in allList:
        index = int(item.split('_')[0])
        allName = './pics/' + item
        temp = handleOnePic(allName)
        # if (index != 1):
        #     continue
        # # print(allName)
        # print(temp)
        temp.append(index)
        data.append(temp)
    
    # for i in data:
    #     print(i)
    # exit()
    np.array(data)
    np.save('./data.npy', data)
    # np.savetxt('./data.txt',data)