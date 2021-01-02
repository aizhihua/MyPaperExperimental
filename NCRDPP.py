import numpy as np
import pandas as pd
import time
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from pandas import DataFrame

# 用于保存初始邻域的样本类
class Point:
    number = -1 # 编号
    innerRadius = 0 # 内半径
    innerNeighbor = [] # 保存内邻域中的样本
    card = 0 # 邻域中对象的个数
    d = -1 # 保存标签
    def __init__(self,number,innerRadius,innerNeighbor,card,d):
        self.number = number
        self.innerRadius = innerRadius
        self.innerNeighbor = innerNeighbor
        self.card = card
        self.d = d

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
        innerRadius = NM -  0.001 * NH # 得到内邻域的半径
        if innerRadius < 0:
            innerRadius = 0
        innerNeighbor = np.where(disMat[i,:]<=innerRadius)[0] # 得到内邻域的样本
        p = Point(i,innerRadius,innerNeighbor,len(innerNeighbor),nowLabel)
        initCover.append(p)
    return  initCover

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

# 计算每个邻域的质量
def calQuality(Oc):
    qc = [] # 保存每个邻域的质量
    pai = 3
    radii = []
    for o in Oc:
        radii.append(o.innerRadius)
    arrRadii = np.array(radii)
    eta = 0.2 * np.median(arrRadii)
    for o in Oc:
        qi = 1/(1+math.exp(-pai*(o.innerRadius-eta)))
        qc.append(qi)
    return np.array(qc)

# 计算邻域之间的相似度
def calSimilarityMatrix(Oc):
    OcSha = len(Oc)
    simMatrix = np.zeros((OcSha,OcSha)) # 保存每个邻域之间的相似度
    for i in range(OcSha):
        for j in range(OcSha):
            if simMatrix[i,j] ==0:
                simMatrix[i, j] = len(np.intersect1d(Oc[i].innerNeighbor,Oc[j].innerNeighbor))/min(len(Oc[i].innerNeighbor),len(Oc[j].innerNeighbor))
                simMatrix[j, i] = simMatrix[i, j]
    return simMatrix

# ---- test ---------------
def elem_sympoly(lmbda, k):
    N = len(lmbda)
    E= np.zeros((k+1,N+1))
    E[0,:] =1
    for l in range(1,(k+1)):
        for n in range(1,(N+1)):
            E[l,n] = E[l,n-1] + lmbda[n-1]*E[l-1,n-1]
    return E

def sample_k_eigenvecs(lmbda, k):
    np.random.seed(0)  # 设置随机种子
    E = elem_sympoly(lmbda, k)
    i = len(lmbda)
    rem = k
    S = []
    while rem>0:
        if i==rem:
            marg = 1
        else:
            marg= lmbda[i-1] * E[rem-1,i-1]/E[rem,i]

        if np.random.random()<marg:
            S.append(i-1)
            rem-=1
        i-=1
    S= np.array(S)
    return S

def perPropess(initCover,train):
    C = set(train[:, -1])
    qcList = []
    scList = []
    for c in C:
        Oc = []
        for neighborhood in initCover:  # 获取同类的邻域集合
            if neighborhood.d == c:
                Oc.append(neighborhood)
        qc = calQuality(Oc)  # 计算邻域质量
        qcList.append(qc)
        Sc = calSimilarityMatrix(Oc)
        scList.append(Sc)
    return qcList,scList


# 邻域覆盖约简Dpp
def coveringReductionDpp(initCover,train,k,qcList, scList):
    np.random.seed(0)
    C = set(train[:,-1])
    Os = [] #保存最后用于分类的邻域集合
    n = len(initCover)
    for j,c in enumerate(C):
        Oc = []
        for neighborhood in initCover: # 获取同类的邻域集合
            if neighborhood.d == c:
                Oc.append(neighborhood)
        nc = len(Oc)
        kc = math.ceil(k*nc/n) # 当前类需要选择的邻域个数
        k = k - kc # 剩余个数
        n = n - nc
        if kc == 0:
            continue
        # qc = calQuality(Oc) # 计算邻域质量
        qc = qcList[j]
        # Sc = calSimilarityMatrix(Oc)
        Sc = scList[j]
        qcT = qc.reshape((nc,1))
        temp = qc * qcT
        L = temp * Sc # 得到L型矩阵
        eigenvalue, featurevector = np.linalg.eigh(L) # 得到特征值 特征向量
        eigenvalue = np.real(eigenvalue)
        featurevector = np.real(featurevector)
        jidx = sample_k_eigenvecs(eigenvalue,kc)
        V = featurevector[:,jidx]
        Y = []
        N = eigenvalue.shape[0]
        for i in range(kc-1,-1,-1):
            P = np.sum(V ** 2, axis=1)
            row_idx = np.random.choice(range(N), p=P / np.sum(P))
            col_idx = np.nonzero(V[row_idx])[0][0]
            Y.append(row_idx)
            # update V
            V_j = np.copy(V[:, col_idx])
            V = V - np.outer(V_j, V[row_idx] / V_j[row_idx])
            V[:, col_idx] = V[:, i]
            V = V[:, :i]
            # orthogonalize
            for a in range(0, i - 1):
                for b in range(0, a):
                    V[:, a] = V[:, a] - np.dot(V[:, a], V[:, b]) * V[:, b]
                norm = np.linalg.norm(V[:, a])
                assert norm > 0
                V[:, a] = V[:, a] / norm
        [Os.append(Oc[i]) for i in Y]
    return Os

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
            toBorderDis = dis - rule.innerRadius
            toBorder.append(toBorderDis)
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
    Ks = [12,28,8,11,22]
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
        qcList, scList = perPropess(initCover, trainData)
        Os = coveringReductionDpp(initCover,trainData,Ks[i-1],qcList, scList) # 得到选择后的邻域#Ks[i-1]
        rules = []
        for o in Os:
            rule = Rule(o.number,o.innerRadius,o.d)
            rules.append(rule)
        end = time.time()
        sumTime  += end-start
        varifyData = pd.read_excel("newsplit/verifyData" + str(i) + ".xlsx").values  # 获取验证数据集
        testData = pd.read_excel("newsplit/testData"+str(i)+".xlsx").values # 获取训练数据集
        true,pre = predict(testData,trainData,rules)
        acc = accuracy_score(true,pre)
        sumAcc += acc
        f1 = f1_score(true, pre, average='macro')
        sumF1 += f1;
    return sumAcc/5,sumF1/5,sum(Ks)/5,sumTime/5



if __name__  == "__main__":
    acc , f1 ,k,fivetime= mainFrame()
    print("time:",fivetime)
    print("acc:",acc)
    print("f1:",f1)
    print("k:",k)
