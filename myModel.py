import pandas as pd
import numpy as np
import time
import copy
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from pandas import DataFrame

#  保存需要后续处理的聚类
class Process:
    lastData = [] # 需要继续处理的数据
    labels = [] # 出现的标签值
    def __init__(self,last,la):
        self.lastData = last
        self.labels = la

# 用于分类的规则类
class Rule:
    x = [] # （中心点坐标）
    innerRadius = 0  # 内半径
    d = -1 # 标签
    card = 0 # 邻域中对象个数（密度）
    def __init__(self,x,innerRadius,d,card):
        self.x = x
        self.innerRadius = innerRadius
        self.d = d
        self.card = card

# 欧式距离计算公式
def euclideanDistance(x,y):
    return np.sqrt(np.sum((x - y)**2))

# kmeans计算下一次的中心
def CreateCenter(items,data):
    nextCreateCent = []
    for item in items:
        cent = []
        subData = data[item,:]
        if(len(subData)!=0):
            cent = list(np.mean(subData,axis=0)) # 下一次中心取均值
        nextCreateCent.append(cent)
    return np.array(nextCreateCent)

# kmeans判断结束条件
def judgeEqual(createCent,nextCreateCent):
    return  (createCent==nextCreateCent).all()

# 计算聚类中心之间的距离
def calCenter(createCent):
    x_sha = len(createCent)
    disMat = np.zeros((x_sha, x_sha), dtype=float)  # 保存样本之间的距离
    for i in range(x_sha):
        for j in range(x_sha):
            if disMat[i, j] == 0:
                disMat[i, j] = euclideanDistance(np.array(createCent[i]), np.array(createCent[j]))
                disMat[j, i] = disMat[i, j]
    return disMat

# kmeans框架
def kmeans(center,data):
    createCent = center # 装每个中心的坐标
    items = [] # 装每个中心的聚类
    tempItems = []
    neibor = [] # 保存聚类收敛后的对象
    radius = [] # 保存半径
    for i in range(len(center)):
        items.append([])
        tempItems.append([])
        neibor.append([])
    while(True):
        dst = np.linalg.norm(data - createCent[:, None], axis=2)
        index = np.argmin(dst, axis=0)
        flag = 0
        for i in index:
            items[i].append(flag)
            flag += 1
        nextCreateCent = CreateCenter(items,data)
        flag = judgeEqual(createCent,nextCreateCent)
        if flag:# 已经中心不变
            disMat = calCenter(nextCreateCent)
            for i in range(len(items)):
                lis = dst[i, :]
                r = np.max(lis[[items[i]]])
                disIndex = np.delete(disMat[i, :], i)
                rtemp = np.min(disIndex)
                if r > rtemp:
                    r = rtemp
                radius.append(r)
            return  radius, np.array(nextCreateCent) # 返回邻域对象，半径，中心点位置
        else: # 变化
            createCent = nextCreateCent
            items = copy.deepcopy(tempItems)

#  计算邻域覆盖
def neighborhoodCovering(train):
    rules = []  # 保存所有的规则集合
    runOnData = [] # 保存需要继续处理的邻域
    le = len(train)
    pro = Process(train, list(set(train[:,-1])))
    runOnData.append(pro)
    all = np.array([])
    while (len(runOnData) != 0):  # 需要继续分解
        runOn = runOnData[0]
        runOnData.remove(runOn)
        center = []  # 保存中心点
        data = runOn.lastData
        for i in runOn.labels:
            temp = data[:, -1] == i
            cent = np.mean(data[temp, :-1], axis=0)
            center.append(cent.tolist())
        radius, cents = kmeans(np.array(center), data[:, :-1])
        for i,cent in enumerate(cents):
            dst = np.linalg.norm(train[:,:-1] - cent, axis=1)
            item = np.where(dst < radius[i])[0]
            temp = list(set(train[item, -1]))
            if len(temp) == 1:  # 全为同类
                rule = Rule( cent,radius[i], temp[0], len(item))  # ,x,innerRadius,d,card
                rules.append(rule)
                all = np.hstack((all,item))
            elif len(temp) > 1:  # 出现异类
                pro = Process(train[item, :], temp)
                runOnData.append(pro)
    all = set(all.tolist())
    rest = np.arange(0,le).tolist()
    [rest.remove(i) for i in all]
    otherRules = []
    for i in rest:
        rule = Rule(train[i,:-1],0, train[i,-1], 1)  # ,x,innerRadius,d,card
        otherRules.append(rule)
    return rules,otherRules

# 预测
def predict(test,rules):
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
            dis = euclideanDistance(row[:-1],rule.x) # 计算测试样本到邻域中心之间的距离
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
def calTime():
    n = 10
    start = time.time()
    for i in range(n):
        mainFrame()
    end = time.time()
    sumTime = end - start
    print(sumTime/(n*5)) # 5折
# 规则选择
def neighborhoodSelection(rules,otherRules):
    sortIndex = []
    num = len(rules)
    new_rules = []
    temp = []
    for rule in rules:
        temp.append(rule.card)
    temp = np.argsort(temp)
    tempRules = []
    [tempRules.append(rules[i]) for i in temp]
    arr = tempRules[0].x
    for i in range(1,num):
        arr = np.vstack((arr,tempRules[i].x))
    avgDis = 0
    for i in range(num-1):
        arr = np.delete(arr, 0, axis=0)
        dst = np.linalg.norm(arr[:,:] - tempRules[i].x, axis=1)
        minDis = np.min(dst)
        avgDis += minDis
        sortIndex.append(-minDis*tempRules[i].card)
    tempRules = tempRules + otherRules
    sortIndex = sortIndex + [-avgDis/len(tempRules)] * len(otherRules)
    sortIndex = np.argsort(sortIndex)
    new_rules.append(tempRules[num-1])
    [new_rules.append(tempRules[i]) for i in sortIndex]
    return new_rules

def verMySelect():
    trainData = pd.read_excel("neighSelect/trainData.xlsx").values  # 获取训练数据集
    rules = neighborhoodCovering(trainData)  # 返回规则集合
    new_rules = neighborhoodSelection(rules)
    testData = pd.read_excel("neighSelect/testData.xlsx").values  # 获取训练数据集
    accList = []
    f1List = []
    rows = []
    for i in range(1,101):
        true, pre = predict(testData, new_rules[:i])  # 预测
        acc = accuracy_score(true, pre)
        f1 = f1_score(true, pre, average='macro')
        accList.append(acc)
        f1List.append(f1)
        rows.append(i)
    accData = DataFrame(data=accList,columns=['FCNC'],index=rows)
    f1Data = DataFrame(data=f1List,columns=['FCNC'],index=rows)
    accData.to_excel("myModelAcc.xlsx")
    f1Data.to_excel("myModelF1.xlsx")

# 主框架
def mainFrame():
    Ks = [69,93,76,74,88]
    sumAcc = 0
    sumF1 = 0
    sumTime = 0
    for i in range(1,6):
        start = time.time()
        trainData = pd.read_excel("newsplit/trainData"+str(i)+".xlsx").values # 获取训练数据集
        rules,otherRules = neighborhoodCovering(trainData) # 返回规则集合
        new_rules = neighborhoodSelection(rules,otherRules)
        end = time.time()
        sumTime += end - start
        testData  = pd.read_excel("newsplit/testData"+str(i)+".xlsx").values # 获取训练数据集
        varifyData = pd.read_excel("newsplit/verifyData"+str(i)+".xlsx").values # 获取验证数据集
        true, pre = predict(varifyData, new_rules[:Ks[i-1]])  # 预测
        acc = accuracy_score(true, pre)
        sumAcc += acc
        f1 = f1_score(true, pre, average='macro')
        sumF1 += f1
    return sumAcc / 5, sumF1 / 5, sum(Ks) / 5, sumTime/5


if __name__ == "__main__":

    acc, f1, k, fivetime = mainFrame()
    # print("time:",fivetime)
    # print("acc:",acc)
    # print("f1:",f1)
    # print("k:",k)
    # verMySelect()
    # calTime()
