import numpy as np
from sklearn import svm
import random
from sklearn.model_selection import KFold
import joblib
import cv2
from PIL import Image, ImageDraw, ImageFont
import math
import pyautogui



#---------------------------识别算法处理部分-----------------------------------

# 加载单个数据集
def loadData(rate):
    data = np.load('data.npy').tolist()
    datas = []
    for i in range(9):
        datas.append([])
    for item in data:
        if int(item[8]) == 11:
            # print('con')
            continue
        # print(int(item[7] - 1))
        datas[int(item[7] - 1)].append(item)
    
    trainData = []
    testData = []
    for item in datas:
        number = int(rate * len(item))
        choise = random.sample(range(len(item)),number)
        for i in range(len(item)):
            if i in choise:
                testData.append(item[i])
            else:
                trainData.append(item[i])

    trainData = np.array(trainData)
    testData = np.array(testData)
    trainX = trainData[:,0:4]
    trainY = trainData[:,-1]
    testX = testData[:,0:4]
    testY = testData[:,-1]
    return trainX, trainY, testX, testY

def loadPureData():
    data = np.load('data.npy')
    return data, data[:,0:4], data[:,-1]



# n倍交叉验证法辅助函数，对于一个数组提取测试集与验证集
def CrossVDataSplit(datas, n):
    datas = np.array(datas)
    kf = KFold(n_splits=n,shuffle=True)
    trainData = []
    testData = []
    for train_index, test_index in kf.split(datas):
        TtrainData = datas[train_index]
        TtestData = datas[test_index]
        trainData.append(TtrainData.tolist())
        testData.append(TtestData.tolist())
    return trainData, testData


# n倍交叉验证法数据生成
def CrossVData(n):
    data = np.load('data.npy').tolist()
    datas = []
    for i in range(9):
        datas.append([])
    for item in data:
        datas[int(item[8] - 1)].append(item)
    datas = np.array(datas)
    trainData = []
    testData = []
    for item in datas:
        item = np.array(item)
        trainD, testD = CrossVDataSplit(item, n)
        trainData.append(trainD)
        testData.append(testD)
    trainDatas = []
    testDatas = []
    # print(np.array(trainData[0][0]).shape)

    for i in range(n):
        tempTrain = np.empty((0,9))
        tempTest = np.empty((0,9))
        for j in range(9):
            tempTrain = np.vstack((tempTrain, np.array(trainData[j][i])))
            tempTest = np.vstack((tempTest, np.array(testData[j][i])))
        trainDatas.append(tempTrain)
        testDatas.append(tempTest)
    return trainDatas,testDatas



# kd树
class kd_tree:
  def __init__(self, value):
    self.value = value
    self.dimension = None  
    self.left = None
    self.right = None
   
  def setValue(self, value):
    self.value = value
   

# kd数搜索，减少搜索时间
def creat_kdTree(dataIn, k, root, deep):
  if(dataIn.shape[0]>0): 
    dataIn = dataIn[dataIn[:,int(deep%k)].argsort()]   
    data1 = None; data2 = None
  
    if(dataIn.shape[0]%2 == 0): 
      mid = int(dataIn.shape[0]/2)
      root = kd_tree(dataIn[mid,:])
      root.dimension = deep%k
      dataIn = np.delete(dataIn,mid, axis = 0)
      data1,data2 = np.split(dataIn,[mid], axis=0) 

    elif(dataIn.shape[0]%2 == 1):
      mid = int((dataIn.shape[0]+1)/2 - 1) 
      root = kd_tree(dataIn[mid,:])
      root.dimension = deep%k
      dataIn = np.delete(dataIn,mid, axis = 0)
      data1,data2 = np.split(dataIn,[mid], axis=0)
    deep+=1
    root.left = creat_kdTree(data1, k, None, deep)
    root.right = creat_kdTree(data2, k, None, deep)
  return root

#k近邻搜索
def findKNode(kdNode, closestPoints, x, k):
  if kdNode == None:
    return
  curDis = (sum((kdNode.value[0:4]-x[0:4])**2))**0.5
  tempPoints = closestPoints[closestPoints[:,9].argsort()]
  for i in range(k):
    closestPoints[i] = tempPoints[i]
  if closestPoints[k-1][9] >=1000000 or closestPoints[k-1][9] > curDis:
    closestPoints[k-1][9] = curDis
    closestPoints[k-1,0:9] = kdNode.value 
  if kdNode.value[kdNode.dimension] >= x[kdNode.dimension]:
    findKNode(kdNode.left, closestPoints, x, k)
  else:
    findKNode(kdNode.right, closestPoints, x, k)
  rang = abs(x[kdNode.dimension] - kdNode.value[kdNode.dimension])
  if rang > closestPoints[k-1][9]:
    return
  if kdNode.value[kdNode.dimension] >= x[kdNode.dimension]:
    findKNode(kdNode.right, closestPoints, x, k)
  else:
    findKNode(kdNode.left, closestPoints, x, k)

# knn中判断哪个类最多
def findMaxIndex(closePoint):
    t = [0 for i in range(9)]
    for item in closePoint:
        # print(item)
        t[int(item[-2]) - 1] += 1
    mIndex = 0
    for i in range(9):
        if t[mIndex] < t[i]:
            mIndex = i
    return mIndex + 1

# 标签one-hot处理
def onehot(targets, num):
    result = np.zeros((num, 9))
    for i in range(num):
        result[i][int(targets[i]) - 1] = 1
    return result

# 逆向one-hot
def reverOneHot(arr):
    for i in range(len(arr)):
        if arr[i] == 1:
            return i + 1
    return i

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid的一阶导数
def Dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

#神经网络模型
class Modle(object):

    #初始化
    def __init__(self, l0 = 10, l1 = 10, l2 = 10, l3 = 10):
        # self.lr = 0.001            # 学习率
        self.lr = 1e-6
        self.W1 = np.random.randn(l0, l1) * 0.1    
        self.b1 = np.random.randn(l1) * 0.1
        self.W2 = np.random.randn(l1, l2) * 0.1
        self.b2 = np.random.randn(l2) * 0.1
        self.W3 = np.random.randn(l2, l3) * 0.1
        self.b3 = np.random.randn(l3) * 0.1
    
    def setW(self, w1, b1, w2, b2, w3, b3):
        self.W1 = w1
        self.b1 = b1
        self.W2 = w2
        self.b2 = b2
        self.W3 = w3
        self.b3 = b3

    # 前向传播
    def forward(self, X, y):
        self.X = X                                     
        self.z1 = np.dot(X, self.W1) + self.b1           
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2       
        self.a2 = sigmoid(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3      
        self.a3 = sigmoid(self.z3)


        loss = np.sum((self.a3 - y) * (self.a3 - y)) / 2  
        self.d3 = (self.a3 - y) * Dsigmoid(self.z3)       
        return loss, self.a3
    
    # 预测，实际上就是前向传播再成one-hot形式
    def predict(self, X):
        self.X = X                                 
        self.z1 = np.dot(X, self.W1) + self.b1      
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2  
        self.a2 = sigmoid(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3        
        self.a3 = sigmoid(self.z3)
        a3 = self.a3


        for i in range(a3.shape[0]):
            mIndex = 0
            for j in range(a3.shape[1]):
                if j == mIndex:
                    continue
                if a3[i,j] > a3[i,mIndex]:
                    a3[i,mIndex] = 0
                    mIndex = j
                else:
                    a3[i,j] = 0
            a3[i,mIndex] = 1

        return a3


    #反向传播
    def backward(self):
        dW3 = np.dot(self.a2.T, self.d3)                 
        db3 = np.sum(self.d3, axis=0)                  

        d2 = np.dot(self.d3, self.W3.T) * Dsigmoid(self.z2)  
        dW2 = np.dot(self.a1.T, d2)                       
        db2 = np.sum(d2, axis=0)                       

        d1 = np.dot(d2, self.W2.T) * Dsigmoid(self.z1)  
        dW1 = np.dot(self.X.T, d1)                       
        db1 = np.sum(d1, axis=0)                        

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

#载入模型权重初值
def loadModel():
    w = np.load('nerve.npz')
    t = Modle()
    t.setW(w['w1'], w['b1'], w['w2'], w['b2'], w['w3'], w['b3'])
    return t


def train(model, trainX, trainY, isSave = 0):
    # 100轮迭代
    for epoch in range(100):
        # 每次迭代3个样本
        for i in range(0, len(trainX), 3):
            X = trainX[i:i + 3]
            y = trainY[i:i + 3]
            loss, _ = model.forward(X, y)
            # print("Epoch:", epoch, "-", i, ":", "{:.3f}".format(loss))  #输出提示
            model.backward()
        # 保存模型
        if epoch % 100 == 0 and isSave:
            np.savez("nerve.npz", w1=model.W1, b1=model.b1, w2=model.W2, b2=model.b2, w3=model.W3, b3=model.b3)
    return model

# 三种方法投票， 都不同选第一种
def vote(arr1, arr2, arr3):
    res = []
    for i in range(len(arr1)):
        if arr2[i] == arr3[i]:
            res.append(arr2[i])
        else:
            res.append(arr1[i])
    return res

def calCorrectRate(arr1, arr2):
    correct = 0
    for i in range(len(arr1)):
        if arr1[i] == arr2[i]:
            correct += 1
    return correct / len(arr1)


class triModel(object):
    def __init__(self):
        trainDatas, trainX, trainY = loadPureData()
        self.svc= joblib.load("svmModel.joblib")
        self.tree = creat_kdTree(trainDatas, 4, None, 0)
        self.model = loadModel()
    
    def predict(self,X):
        X = X.reshape((1,-1))
        # svm
        svmTest = self.svc.predict(X.reshape((1,-1)))[0]

        # knn
        closePoint = np.zeros((3,10))
        closePoint[:,9] = 100000
        findKNode(self.tree, closePoint, X[0], 3)
        knnTest = findMaxIndex(closePoint)

        #神经网络
        nerveTest = self.model.predict(X)
        nerveTest = reverOneHot(nerveTest[0])

        return vote([nerveTest], [svmTest], [knnTest])[0]

#---------------------------opencv处理部分-----------------------------------

# 肤色检测
def skinFind(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) # 把图像转换到YUV色域
    x = 140
    y = 77
    skin3 = cv2.inRange(ycrcb,(0,x,y),(255,x + 33,y + 63))
    return skin3

# 处理窗口函数乱码
def winname(name):
    return name.encode("gbk").decode(errors="ignore")

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

def handleOnePic(frame):
    #读取图片
    # frame = cv2.imread(picName)
    # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
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
    if len(maxCount) == 0:
        return 0
    area = cv2.contourArea(maxCount) / frame.size
    if (area < 0.01):
        return 0
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
    res = [] #返回的数据 分别为： 峰值个数 到掌心距离的平均值   峰值与手掌方向夹角的总和 峰值到掌心距离总和 掌心x 掌心y 手掌方向斜率 图像占整个图像面积的比
    
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
        totalDis += dis / maxDistance * 100
    
    res.append(angles)
    res.append(totalDis)
    res.append(x3)
    res.append(y3)
    res.append(math.atan2(stander[1], stander[0]))
    res.append(area)
    return res

# opencv往图片中写入中文,返回图片，直接用opencv写中文会乱码
def DrawChinese(img, text, angle = '',size = 25, X = 160):
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)  
    font = ImageFont.truetype("simhei.ttf", 18, encoding="utf-8") 
    x = 130
    y=100
    w=120
    h=80
    r=20
    color=(255,0,0)
    draw.ellipse((x,y,x+r,y+r),fill=color)    
    draw.ellipse((x+w-r,y,x+w,y+r),fill=color)    
    draw.ellipse((x,y+h-r,x+r,y+h),fill=color)    
    draw.ellipse((x+w-r,y+h-r,x+w,y+h),fill=color)
    draw.rectangle((x+r/2,y, x+w-(r/2), y+h),fill=color)    
    draw.rectangle((x,y+r/2, x+w, y+h-(r/2)),fill=color)
    draw.text((140,105), '预测手势为：', (255,255,255), font=font)
    font = ImageFont.truetype("simhei.ttf", 16, encoding="utf-8") 
    draw.text((141,157), angle, (255,255,255), font=font)
    font = ImageFont.truetype("simhei.ttf", size, encoding="utf-8")  
    draw.text((X,127), text, (255,255,255), font=font)
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR) 
    return cv2charimg

def drawRect(draw, x, y, w, h, r):
    # x = 10
    # y=80
    # w=100
    # h=50
    # r=20
    color=(255,0,0)
    draw.ellipse((x,y,x+r,y+r),fill=color)    
    draw.ellipse((x+w-r,y,x+w,y+r),fill=color)    
    draw.ellipse((x,y+h-r,x+r,y+h),fill=color)    
    draw.ellipse((x+w-r,y+h-r,x+w,y+h),fill=color)
    draw.rectangle((x+r/2,y, x+w-(r/2), y+h),fill=color)    
    draw.rectangle((x,y+r/2, x+w, y+h-(r/2)),fill=color)
    return draw


# 猜拳游戏的图像文字，直接用opencv写中文会乱码
def DrawChinese2(img, text = "拳头", text1 = "拳头", text3 = "平局"):
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)  
    font = ImageFont.truetype("simhei.ttf", 25, encoding="utf-8")
    draw = drawRect(draw, 10, 80, 100, 50, 20)
    draw = drawRect(draw, 500, 80, 100, 50, 20)
    draw = drawRect(draw, 250, 10, 100, 50, 20)
    draw.text((30, 90), text, (255,255,255), font=font)
    draw.text((520, 90), text1, (255,255,255), font=font)
    draw.text((270, 20), text3, (255,255,255), font=font)
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR) 
    return cv2charimg

#判断数组变化方式
def findDir(arr):
    increase = 0
    temp = -11.11
    for item in arr:
        if temp == -11.11:
            temp = item
            continue
        if temp < item :
            increase += 1
        if temp > item :
            increase -= 1
        temp = item
    if increase > 0:
        return 1
    else:
        return 0


def between(a, b, c):
    if a > b and a < c:
        return 1
    return 0

# 动态手势识别检测
def checkMov(arr):
    # print(arr[0])
    # arr = [[69.0, 185.0, -1.6371442733533041, 0.08371666666666666, 1.0], [72.0, 186.0, -1.648109750786073, 0.08211666666666667, 1.0], [77.0, 192.0, -1.694374715826205, 0.08425555555555556, 1.0], [80.0, 187.0, -1.6301877500647837, 0.08538888888888889, 1.0], [85.0, 187.0, -1.627886639167759, 0.08497222222222223, 1.0], [92.0, 189.0, -1.7077057312073443, 0.08445555555555556, 1.0], [97.0, 190.0, -1.6854299824000147, 0.08466111111111112, 1.0], [104.0, 189.0, -1.6609059479082877, 0.08482222222222222, 1.0], [112.0, 189.0, -1.677343719619815, 0.08721111111111111, 1.0], [119.0, 190.0, -1.6650304412349286, 0.08407777777777778, 1.0], [126.0, 193.0, -1.6732728288513614, 0.08783888888888888, 1.0], [136.0, 188.0, -1.7056789887618649, 0.08477222222222222, 1.0], [148.0, 193.0, -1.7313579294970711, 0.08840555555555556, 1.0], [159.0, 195.0, -1.7049301794075202, 0.09046111111111112, 1.0], [170.0, 188.0, -1.7430455375411587, 0.08969444444444444, 1.0], [182.0, 198.0, -1.6785181237990199, 0.08701111111111111, 1.0], [194.0, 201.0, -1.6086604277804275, 0.08621666666666666, 1.0], [207.0, 192.0, -1.7110573996305718, 0.08765555555555556, 1.0], [219.0, 194.0, -1.716024722596257, 0.08766666666666667, 1.0], [229.0, 193.0, -1.7746056359462417, 0.08908888888888888, 1.0]]
    handList = np.array(arr)
    tempList = np.empty((0,5))
    for item in handList:
        if len(tempList) == 0:
            tempList = np.vstack((tempList, item))
            continue
        if tempList[0][-1] == item[-1]:
            tempList = np.vstack((tempList, item))
        else:
            if len(tempList) > 10:
                break
            else:
                tempList = np.empty((0,5))
    if len(tempList) == 0:
        return '无'

    areas = tempList[:,3]
    # print(areas)
    areasChange = findDir(areas)
    # print(areas)
    if np.max(areas) - np.min(areas) > 0.06:
        if areasChange:
            return '数字%d推' % tempList[0][-1]
        else:
            return '数字%d拉' % tempList[0][-1]
    
    
    x = tempList[:,0]
    y = tempList[:,1]

    if np.max(x) - np.min(x) < 50 and np.max(y) - np.min(y) < 50:
        return '数字%d无移动' % tempList[0][-1]

    xMean = np.mean(x)
    yMean = np.mean(y)
    xChange = findDir(x)
    yChange = findDir(y)

    b = np.sum(np.multiply((x - xMean), (y - yMean))) / np.sum(np.multiply((x - xMean), (x - xMean)))

    angle = (math.atan(b) * 180 / 3.14)

    

    if  between(angle, -15, 15):
        if xChange == 1:
            return  '数字%d右滑动' % tempList[0][-1]
        else:
            return  '数字%d左滑动' % tempList[0][-1]
    elif between(angle, -65, -30):
        if xChange == 1:
            return  '数字%d右上滑动' % tempList[0][-1]
        else:
            return  '数字%d左下滑动' % tempList[0][-1]
    elif between(angle, 30, 65):
        if xChange == 1:
            return  '数字%d右下滑动' % tempList[0][-1]
        else:
            return  '数字%d左上滑动' % tempList[0][-1]
    else:
        if yChange == 1:
            return  '数字%d下滑动' % tempList[0][-1]
        else:
            return  '数字%d上滑动' % tempList[0][-1]
    return angle
    # print(angle)

#判断胜负
def findWin(res1, res2):
    #2---剪刀   5----布   9----石头
    if res1 == 2:
        text1 = "剪刀"
    elif res1 == 5:
        text1 = "布"
    elif res1 == 9:
        text1 = "石头" 
    else:
        text1 = "无"

    if res2 == 2:
        text2 = "剪刀"
    elif res2 == 5:
        text2 = "布"
    elif res2 == 9:
        text2 = "石头" 
    else:
        text2 = "无"
    
    if text1 == "无" or text2 == "无":
        return text1, text2, "无"
    
    win = 0 #胜者
    if res1 == 2:
        if res2 == 5:
            win = 1
        elif res2 == 9:
            win = 2
        else:
            win = 0
    elif res1 == 5:
        if res2 == 5:
            win = 0
        elif res2 == 9:
            win = 1
        else:
            win = 2
    elif res1 == 9:
        if res2 == 5:
            win = 2
        elif res2 == 9:
            win = 0
        else:
            win = 1
    if win == 1:
        win = "左方胜"
    elif win == 2:
        win = "右方胜"
    else:
        win = "平局"
    return text1, text2, win
    




#主函数
if __name__ == "__main__":
    # 自己的模型
    model = triModel()

    k = input('输入数字：\n1-------静态手势检测\n2-------动态手势检测\n3-------猜拳小游戏\n4-------控制ppt\n')
    k = int(k)
    camera = cv2.VideoCapture(1) #调用摄像头
    handList = []  #用于储存一段时间的手势数据
    handDelay = 0 # 控制ppt时的控制间隔
    while camera.isOpened():
        if k == 1 or k == 2 or k == 4:
            ret, frame = camera.read()
            frame = cv2.flip(frame, 1)  # 水平翻转图片
            frame = cv2.bilateralFilter(frame, 5, 50, 100)  # 双边滤波
            temp = skinFind(frame) #肤色检测
            cX =int(0.45 * frame.shape[1]) #检测区域起始位置
            cY =int(0.2 * frame.shape[0])
            cv2.rectangle(temp, (cX, cY),(cX + 300, cY + 300), (255, 255, 0), 2) #肤色检测后的全图
            # cv2.imshow(winname('图片'),temp)
            cv2.rectangle(frame, (cX, cY),(cX + 300, cY + 300), (255, 255, 0), 2) #显示原图
            #取要检测部分进行处理
            img = temp[cY:cY + 300,cX:cX + 300] 
            img = cv2.medianBlur(img, 5)  # 中值滤波
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) #开运算
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) #闭运算
            cv2.namedWindow(winname('实图'))
            cv2.imshow(winname('实图'), img)
            handleRes = handleOnePic(img)
            cv2.namedWindow(winname('原图'))

        if k == 1:
            if handleRes == 0:  #没有图像的情况下
                cv2.imshow(winname('原图'), DrawChinese(frame,'无'))
            else:
                handleRes = np.array(handleRes)
                
                res = model.predict(handleRes[0:4])
                if res == 9: #手势9为数字0
                    res = 0
                cv2.imshow(winname('原图'), DrawChinese(frame,'数字%d' % res, '方向%.2f度' % ((-handleRes[-2])*180/3.1415)))

        elif k == 2:
            if handleRes == 0:  #没有图像的情况下
                cv2.imshow(winname('原图'), DrawChinese(frame,'无'))
            else:
                handleRes = np.array(handleRes)
                # print(handleRes)
                res = model.predict(handleRes[0:4])
                if res == 9: #手势9为数字0
                    res = 0
                t = handleRes[4:].tolist()
                t.append(res)
                handList.append(t)
                if (len(handList) > 20):
                    handList.pop(0)
                # print(checkMov(handList))
                cv2.imshow(winname('原图'), DrawChinese(frame,checkMov(handList),'',18, 140))
        
        elif k == 3:
            ret, frame = camera.read()
            frame = cv2.flip(frame, 1)  # 水平翻转图片
            frame = cv2.bilateralFilter(frame, 5, 50, 100)  # 双边滤波
            temp = skinFind(frame) #肤色检测
            cX =int(0.02 * frame.shape[1]) #检测区域起始位置
            cY =int(0.3 * frame.shape[0])
            cX1 =int(0.5 * frame.shape[1]) #检测区域起始位置
            cY1 =int(0.3 * frame.shape[0])

            # cv2.imshow(winname('图片'),temp)
            cv2.rectangle(frame, (cX, cY),(cX + 300, cY + 300), (255, 255, 0), 2) #显示原图
            cv2.rectangle(frame, (cX1, cY1),(cX1 + 300, cY1 + 300), (255, 255, 0), 2) #显示原图
            #取要检测部分进行处理
            img = temp[cY:cY + 300,cX:cX + 300] 
            img = cv2.medianBlur(img, 5)  # 中值滤波
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) #开运算
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) #闭运算
            cv2.namedWindow(winname('left图'))
            cv2.imshow(winname('left图'), img)
            handleRes1 = handleOnePic(img)

            img = temp[cY1:cY1 + 300,cX1:cX1 + 300] 
            img = cv2.medianBlur(img, 5)  # 中值滤波
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) #开运算
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) #闭运算
            cv2.namedWindow(winname('right图'))
            cv2.imshow(winname('right图'), img)
            # 处理两个图片
            handleRes2 = handleOnePic(img)

            cv2.namedWindow(winname('原图'))

            if handleRes1 == 0 or handleRes2 == 0:  #没有图像的情况下
                cv2.imshow(winname('原图'), DrawChinese2(frame,'无',"无","无"))
            
            else:
                handleRes1 = np.array(handleRes1)
                handleRes2 = np.array(handleRes2)
                res1 = model.predict(handleRes1[0:4])
                res2 = model.predict(handleRes2[0:4])

                a,b,c = findWin(res1, res2)

                #判断胜负
                cv2.imshow(winname('原图'), DrawChinese2(frame,a,b,c)) #findWin(res1, res2)

        
        elif k == 4:
            if handleRes == 0:  #没有图像的情况下
                cv2.imshow(winname('原图'), DrawChinese(frame,'无'))
            else:
                handleRes = np.array(handleRes)
                # print(handleRes)
                res = model.predict(handleRes[0:4])
                if res == 9: #手势9为数字0
                    res = 0
                t = handleRes[4:].tolist()
                t.append(res)
                handList.append(t)
                if (len(handList) > 20):
                    handList.pop(0)
                # print(checkMov(handList))
                handRes = checkMov(handList)
                cv2.imshow(winname('原图'), DrawChinese(frame,handRes,'',18, 140))
                handDelay += 1
                if handDelay > 20:
                    print(handRes)
                    # print(handRes.find('右滑动'))
                    if handRes.find('右滑动') != -1:
                        handDelay = 0
                        pyautogui.press('right')
                    elif handRes.find('左滑动') != -1:
                        handDelay = 0
                        pyautogui.press('left')
                    elif handRes.find('推') != -1:
                        handDelay = 0
                        pyautogui.scroll(500)
                    elif handRes.find('拉') != -1:
                        handDelay = 0
                        pyautogui.scroll(-500)
        else:
            break
                        

        key =  cv2.waitKey(20)
        if key == 27:  # esc退出
            camera.release()
            cv2.destroyAllWindows()

            


        
        

        
    

    





