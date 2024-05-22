from numpy import *
import numpy as np
from functools import reduce
import csv
import pandas as pd
 

#数据提取
def loadDataSet():
    df = pd.read_csv('train.csv')
    df.isnull().sum()    #缺失值处理
    df.Age = df.Age.fillna(df.Age.mean())

    classVec = np.array(df['Survived']).tolist()        #提取出存活结果的列表

    df.drop(['PassengerId','Survived','Name','Cabin','Embarked'], axis = 1, inplace = True)     #去掉了仓位cabin
    wordsList = df.values.tolist()          #提取属性列表

    return wordsList, classVec
 
#数据查重
def remove_repeat(docList):
    a = list(reduce(lambda x, y: set(x) | set(y), docList))     #一直取列表元素的并集，保证不会有两个人的信息重复
    return a
 
 #数据统计
 #统计每行每个属性的出现次数，返回数组
def statistics(vecList, inputWords):
    resultVec = [0] * len(vecList)
    for word in inputWords:     #检测到对应属性就+1
        if word in vecList:
            resultVec[vecList.index(word)] += 1
 
    return array(resultVec)
 
 
#计算计算每个词在p0v非分类和p1v分类上出现的概率，然后训练
def trainNB(trainMatrix, trainClass):
    # 行数，列数
    numTrainClass = len(trainClass)
    numWords = len(trainMatrix[0])
 
    # 全部都初始化为1，避免出现概率为0的情况出现
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Words = 2.0
    p1Words = 2.0

    # 从第一行开始按行遍历，如果当前类别为1，那么当前属性矩阵行数据与p1Num相加；如果当前类别为0，否则和p0Num相加
    # pwords用于统计两个类别下属性的个数
    for i in range(numTrainClass):
        if trainClass[i] == 1:
            p1Num += trainMatrix[i]
            p1Words += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Words += sum(trainMatrix[i])
    # 计算每种类型里面， 每个属性出现的概率数组
    p0Vec = log(p0Num / p0Words)
    p1Vec = log(p1Num / p1Words)
    # 计算在类别中1出现的概率
    pClass1 = sum(trainClass) / float(numTrainClass)
    return p0Vec, p1Vec, pClass1
 
 
def classifyNB(testVec, p0Vec, p1Vec, pClass1):
    # 朴素贝叶斯分类, max(p0， p1)作为推断的分类
    # 贝叶斯分类主要是通过比较概率，不需要确切数据。ln(x)的函数单调递增，直接用对数比较可以简化计算
    p1 = sum(testVec * p1Vec) + log(pClass1)
    p0 = sum(testVec * p0Vec) + log(1 - pClass1)
    if p0 > p1:
        return 0
    return 1
 
 
def printClass(words, testClass):
    if testClass == 1:
        print(words, '预测存活')
    else:
        print(words, '预测死亡')
 
 
if __name__ == '__main__':

    docList, classVec = loadDataSet()

    allWordsVec = remove_repeat(docList)

    trainMat = list(map(lambda x: statistics(allWordsVec, x), docList))

    p0V, p1V, pClass1 = trainNB(trainMat, classVec)
    
    # 测试数据集
    df = pd.read_csv('test.csv')
    df.drop(['PassengerId','Name','Cabin','Embarked'], axis = 1, inplace = True)
    survived_result = []
    for i in range(290):    #418
        testWords = df.iloc[i].tolist()
        testVec = statistics(allWordsVec, testWords) # 转换成单词向量
        testClass = classifyNB(testVec, p0V, p1V, pClass1) #根据贝叶斯公式，比较各个类别的后验概率，判断当前数据的分类情况
        print(i,end="")
        printClass(testWords, testClass)
        survived_result.append(testClass)
        print('',end="")

    survived_correct=[]
    df = pd.read_csv('test.csv')
    survived_correct = np.array(df['Survived']).tolist()        #提取出存活结果的列表
    #精确率、召回率的计算
    TP_val=0
    TN_val=0
    FP_val=0
    FN_val=0
    for i in range(len(survived_correct)-1):
        if survived_correct[i]==1 and survived_result[i]==1:
            TP_val=TP_val+1
        if survived_correct[i]==0 and survived_result[i]==0:
            TN_val=TN_val+1
        if survived_correct[i]==0 and survived_result[i]==1:
            FP_val=FP_val+1
        if survived_correct[i]==1 and survived_result[i]==0:
            FN_val=FN_val+1

    print('TP')
    print(TP_val)
    print('TN')
    print(TN_val)
    print('FP')
    print(FP_val)
    print('FN')
    print(FN_val)