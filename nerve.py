import numpy as np
import random
from sklearn.model_selection import KFold #交叉验证生成
import matplotlib.pyplot as plt

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

#数据加载
def loadData(rate):
    data = np.load('data.npy').tolist()
    datas = []
    for i in range(10):
        datas.append([])
    for item in data:
        datas[int(item[8] - 1)].append(item)
    
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
    for i in range(n):
        tempTrain = np.empty((0,9))
        tempTest = np.empty((0,9))
        for j in range(9):
            tempTrain = np.vstack((tempTrain, np.array(trainData[j][i])))
            tempTest = np.vstack((tempTest, np.array(testData[j][i])))
        trainDatas.append(tempTrain)
        testDatas.append(tempTest)
    return trainDatas,testDatas

#神经网络模型
class Modle(object):

    #初始化
    def __init__(self, l0 = 10, l1 = 10, l2 = 10, l3 = 10):
        # self.lr = 0.001            # 学习率
        self.lr = 0.22e-4
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
    
    # 预测，实际上就是前向传播再处理一下
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

def loadModel():
    w = np.load('nerve.npz')
    t = Modle()
    t.setW(w['w1'], w['b1'], w['w2'], w['b2'], w['w3'], w['b3'])
    return t

def checkModel(model, testX, testY):
    predictY = model.predict(testX)
    correct = 0
    for i in range(len(predictY)):
        if reverOneHot(predictY[i]) == reverOneHot(testY[i]):
            correct += 1
    return correct / len(predictY)
    # print(correct / len(predictY))

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

def train(model, trainX, trainY, isSave = 0):
    # 100轮迭代
    for epoch in range(100):
        # 每次迭代3个样本
        for i in range(0, len(trainX), 3):
            X = trainX[i:i + 3]
            y = trainY[i:i + 3]
            loss, _ = model.forward(X, y)
            print("Epoch:", epoch, "-", i, ":", "{:.3f}".format(loss))  #输出提示
            model.backward()
        # 保存数据
        if epoch % 100 == 0 and isSave:
            np.savez("nerve.npz", w1=model.W1, b1=model.b1, w2=model.W2, b2=model.b2, w3=model.W3, b3=model.b3)
    return model

if __name__ == '__main__':
    

    trainDatas,testDatas = CrossVData(10)
    cListOfTrain = [] # 交叉验证训练集的准确率数组
    correctList = [] # 交叉验证测试集的准确率数组
    model = Modle(4, 80, 180, 9) #四层神经网络为 4 80 180 9
    for j in range(len(trainDatas)):
        print('第%d轮' % (j + 1))
        trainX = trainDatas[j][:,0:4]
        trainY = trainDatas[j][:,-1]
        testX = testDatas[j][:,0:4]
        testY = testDatas[j][:,-1]

        trainY = onehot(trainY, len(trainY))
        testY = onehot(testY, len(testY))

        model = loadModel()
        model = train(model, trainX, trainY)
        testScore = checkModel(model, testX, testY)
        trainScore = checkModel(model, trainX, trainY)

        correctList.append(testScore)
        cListOfTrain.append(trainScore)
        print("训练集:", trainScore)
        print("测试集:", testScore)
    
    print("测试集平均准确率为：", np.array(cListOfTrain).mean())
    print("训练集平均准确率为：", np.array(correctList).mean())
    showPlt(correctList, cListOfTrain)


