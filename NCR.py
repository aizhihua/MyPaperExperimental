import numpy as np
import pandas as pd
import time
import copy
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from pandas import DataFrame
# 用于保存初始邻域的样本类
class Point:
    number = -1 # 编号
    innerRadius = 0 # 内半径
    innerNeighbor = [] # 保存内邻域中的样本
    card = 0 # 邻域中对象的个数
    def __init__(self,number,innerRadius,innerNeighbor,card):
        self.number = number
        self.innerRadius = innerRadius
        self.innerNeighbor = innerNeighbor
        self.card = card

# 用于分类的规则类
class Rule:
    number = -1 # 编号
    innerRadius = 0  # 内半径
    d = -1 # 标签
    def __init__(self,number,innerRadius,d):
        self.number = number
        self.innerRadius = innerRadius
        self.d = d

# 欧式距离计算公式
def euclideanDistance(x,y):
    return np.sqrt(np.sum((x - y)**2))

# 计算每个样本之间的欧式距离
def distanceMatrix(train):
    x_sha = len(train)
    disMat = np.zeros((x_sha,x_sha),dtype=float) # 保存样本之间的距离
    for i in range(x_sha):
        for j in range(x_sha):
            if disMat[i,j] == 0:
                disMat[i, j] = euclideanDistance(train[i,:],train[j,:])
                disMat[j, i] = disMat[i, j]
    return disMat

# 计算初始邻域覆盖
def calInitCovering(train,disMat):
    x_sha = len(train)
    initCover = [] # 用于保存所有样本形成的邻域（覆盖）
    for i in range(x_sha):
        nowLabel = train[i, -1]  # 当前样本的标签
        sameIndex = np.where(train[:,-1] == nowLabel)[0]  # 获取同类点的坐标
        diffIndex = np.where(train[:,-1] != nowLabel)[0]  # 获取异类点的坐标
        sameIndex = np.delete(sameIndex,np.where(sameIndex==i)[0][0]) # 删除本身，为后续计算最近的同类距离
        NH = np.min(disMat[i,sameIndex]) # 最近同类的距离
        NM = np.min(disMat[i,diffIndex]) # 最近的异类距离
        innerRadius = NM -  NH # 得到内邻域的半径
        if innerRadius < 0:
            innerRadius = 0
        innerNeighbor = np.where(disMat[i,:]<=innerRadius)[0] # 得到内邻域的样本
        p = Point(i,innerRadius,innerNeighbor,len(innerNeighbor))
        initCover.append(p)
    return  initCover

# 判断s1是否为s2的子集
def judge(s1,s2):
    for j in s1:
        if j not in s2:
            return False
    return True

# 邻域覆盖约简
def coveringReduction(initCover,train):
    rules = [] # 保存最后的分类规则集合
    while len(initCover) != 0:
        maxlen = 0
        maxp = 0
        for i,p in enumerate(initCover): # 找到覆盖样本最多的邻域
            if p.card > maxlen:
                maxlen = p.card
                maxp = p
        rule = Rule(maxp.number,maxp.innerRadius,train[maxp.number,-1])
        rules.append(rule)
        delNeighbor = [] # 保存待删除的邻域
        for p in initCover:
            if judge(p.innerNeighbor,maxp.innerNeighbor) :
                delNeighbor.append(p)
        [initCover.remove(p) for p in delNeighbor]
    return rules

# 预测
def predict(test,train,rules):
    true = [] # 保存测试样本的真实标签
    pre = [] # 保存测试样本的预测标签
    for row in test:
        trueLabel = row[-1]
        true.append(trueLabel)
        preLabel = -1
        toBorder = []
        proDis = []
        proRules = []
        for rule in rules:
            dis = euclideanDistance(row[:-1],train[rule.number,:-1]) # 计算测试样本到邻域中心之间的距离
            toBorder.append(dis)
            if dis <= rule.innerRadius:
                proDis.append(dis)
                proRules.append(rule)
        if len(proDis)!= 0:  # 测试样本存在与某个邻域之中
            tempArr = np.array(proDis)
            index = np.where(tempArr == np.min(tempArr))[0][0]
            preLabel = proRules[index].d
            pre.append(preLabel)
        if preLabel == -1: # 不存在任意邻域之中
            tempArr =  np.array(toBorder)
            index = np.where(tempArr == np.min(tempArr))[0][0]
            preLabel = rules[index].d
            pre.append(preLabel)
    return true,pre




# 算法主框架
def mainFrame():
   # 取前k个规则进行预测
    Ks = [2,5,9,10,31] # 最优k值
    sumAcc = 0
    sumF1 = 0
    sumTime = 0
    for i in range(1,6):
        start = time.time()
        trainData = pd.read_excel("newsplit/trainData"+str(i)+".xlsx").values # 获取训练数据集
        disMat = distanceMatrix(trainData[:,:-1]) # 计算距离矩阵
        # print("距离矩阵计算完毕")
        initCover = calInitCovering(trainData,disMat) # 计算初始邻域覆盖
        # print("初始邻域覆盖计算完毕")
        temp = copy.deepcopy(initCover)
        rules = coveringReduction(temp,trainData) # 约简后得到分类规则
        end = time.time()
        sumTime += end - start
        varifyData = pd.read_excel("newsplit/verifyData" + str(i) + ".xlsx").values  # 获取验证数据集
        testData = pd.read_excel("newsplit/testData"+str(i)+".xlsx").values # 获取训练数据集
        true ,pre = predict(testData,trainData,rules[:Ks[i-1]])
        acc = accuracy_score(true,pre)
        sumAcc += acc
        f1 = f1_score(true, pre, average='macro')
        sumF1 += f1;
    return sumAcc/5,sumF1/5,sum(Ks)/5 ,sumTime/5

if __name__  == "__main__":

    acc ,f1 ,k ,fivetime= mainFrame()
    print("time:",fivetime)
    print("acc:",acc)
    print("f1:",f1)
    print("k:",k)
