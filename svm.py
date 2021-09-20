import numpy as np
from sklearn.svm import LinearSVC
from sklearn import svm
import random
from sklearn.model_selection import KFold #交叉验证生成
import joblib #svm模型保存
import matplotlib.pyplot as plt


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

def showCorrectRate(testY, predictY):
    print(testY)
    print(predictY)
    res = np.zeros((9,2))
    for i in range(len(testY)):
        res[int(testY[i]) - 1,0] += 1
        if int(testY[i]) == int(predictY[i]):
            res[int(testY[i]) - 1,1] += 1
    return res

def findMax(arr):
    maxIndex = 0
    for i in range(len(arr)):
        if arr[maxIndex] < arr[i]:
            maxIndex = i
    return maxIndex

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
    # trainX, trainY, testX, testY = loadData(0.1)
    # print(testX)
    # print(testY)
    # trainDatas,testDatas = CrossVData(10)
    # print(testDatas[0])

    trainDatas,testDatas = CrossVData(10)

    print(trainDatas[0][0])
    cListOfTrain = [] # 交叉验证训练集的准确率数组
    correctList = [] # 交叉验证测试集的准确率数组
    svcList = [] # svc模型数组
    for j in range(len(trainDatas)):
        print('第%d轮' % (j + 1))
        trainX = trainDatas[j][:,0:4]
        trainY = trainDatas[j][:,-1]
        testX = testDatas[j][:,0:4]
        testY = testDatas[j][:,-1]

        svc=svm.SVC(C=100.0, kernel='poly', degree= 2, coef0 = 0.2)
        svc.fit(trainX,trainY)

        testScore = svc.score(testX,testY)
        trainScore = svc.score(trainX,trainY)
        print("训练集:", trainScore)
        print("测试集:", testScore)
        correctList.append(testScore)
        cListOfTrain.append(trainScore)
        svcList.append(svc)

    print("测试集平均准确率为：", np.array(cListOfTrain).mean())
    print("训练集平均准确率为：", np.array(correctList).mean())
    # print(svcList[0].predict(testDatas[0][:,0:4]), testDatas[0][:,-1])
    bestSvc = svcList[findMax(correctList)]

    joblib.dump(bestSvc, "svmModel.joblib")
    showPlt(correctList, cListOfTrain)

    

    





