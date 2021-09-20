import numpy as np
import random
from sklearn.model_selection import KFold #交叉验证生成
import matplotlib.pyplot as plt



def loadData(rate):
    data = np.load('data.npy').tolist()
    datas = []
    for i in range(10):
        datas.append([])
    for item in data:
        if int(item[7]) == 11:
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
    # print(trainData)
    # print(testData)
    trainData = np.array(trainData)
    testData = np.array(testData)
    trainX = trainData[:,0:4]
    trainY = trainData[:,-1]
    testX = testData[:,0:4]
    testY = testData[:,-1]
    return trainX, trainY, testX, testY

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


# n倍交叉验证法
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

#考虑到不同特征值范围差别很大的影响，可对这类数据进行最大最小值标准化数据集
def normData(dataset):
    maxVals = dataset.max(axis=0) #求列的最大值
    minVals = dataset.min(axis=0) #求列的最小值
    ranges = maxVals - minVals
    retData = (dataset - minVals) / ranges
    return retData, minVals, ranges

# 排序
def sort(arr, arr2):
    for i in range(len(arr)):
        for j in range(i, len(arr)):
            if arr[i] > arr[j]:
                t = arr[i]
                arr[i] = arr[j]
                arr[j] = t
                t = arr2[i]
                arr2[i] = arr2[j]
                arr2[j] = t
    return arr,arr2


# 原始knn算法
def knn(trainX, trainY, testX, k):
    t = testX
    tY = trainY.copy()
    for i in range(1,trainX.shape[0]):
        t = np.vstack((t, testX))
    distSquareMat = (t - trainX) ** 2 
    distSquareSums = distSquareMat.sum(axis=1)
    distances = distSquareSums ** 0.5
    distances, tY = sort(distances,tY)
    res = np.zeros((10))
    for i in range(k):
        res[int(tY[i]) - 1] += 1
    t = 0
    for i in range(10):
        if res[t] < res[i]:
            t = i
    return t + 1


class kd_tree:
  def __init__(self, value):
    self.value = value
    self.dimension = None  
    self.left = None
    self.right = None
   
  def setValue(self, value):
    self.value = value

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

# 交叉验证柱状图
def showPlt(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    bar_width = 0.45
    name_list = [i for i in range(1,11)]
    plt.bar(np.arange(10), 1- arr1, label='测试集',width = bar_width)
    plt.bar(np.arange(10)+bar_width, 1 - arr2, tick_label = name_list,label='训练集', width = bar_width)
    plt.title(u'交叉验证错误率')
    plt.legend()
    plt.show()

    




#主函数
if __name__ == "__main__":
    trainDatas,testDatas = CrossVData(10)
    correctList = [0 for i in range(10)]
    trainCorrectList = [0 for i in range(10)]

    for j in range(10):
        print('第%d轮'% (j + 1))

        myTree = creat_kdTree(trainDatas[j], 4, None, 0)
        
        for i in range(len(testDatas[j])):
            closePoint = np.zeros((3,10))
            closePoint[:,9] = 100000
            # print(closePoint)
            findKNode(myTree,closePoint,testDatas[j][i],3)
            # print(closePoint)
            if findMaxIndex(closePoint) == testDatas[j][i,-1]:
                correctList[j] += 1
        correctList[j] = correctList[j] / len(testDatas[j])

        for i in range(len(trainDatas[j])):
            closePoint = np.zeros((3,10))
            closePoint[:,9] = 100000
            findKNode(myTree,closePoint,trainDatas[j][i],3)
            # print(closePoint)
            if findMaxIndex(closePoint) == trainDatas[j][i,-1]:
                trainCorrectList[j] += 1
        trainCorrectList[j] = trainCorrectList[j] / len(trainDatas[j])

        print("训练集:", trainCorrectList[j])
        print("测试集:", correctList[j])
    
    print("测试集平均准确率为：", np.array(correctList).mean())
    print("训练集平均准确率为：", np.array(trainCorrectList).mean())
    showPlt(correctList, trainCorrectList)
