import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from sklearn import metrics
from sklearn.metrics import auc 
import matplotlib.pyplot as plt
 
if __name__ == '__main__':

    #加载数据
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    #数据预处理
    train_data['Age'].fillna(train_data['Age'].mean(),inplace = True)
    test_data['Age'].fillna(test_data['Age'].mean(),inplace = True)

    #特征选择
    features = ['Pclass','Sex','Age','SibSp','Parch']
    train_features = train_data[features]
    train_labels = train_data['Survived']
    test_features = test_data[features]

    #将符号0-1化
    dv = DictVectorizer(sparse = False)
    train_features = dv.fit_transform(train_features.to_dict(orient= 'records'))

    #建立决策树并进行训练
    clf = DecisionTreeClassifier(criterion= 'entropy')
    clf.fit(train_features,train_labels)

    #预测
    test_features = dv.transform(test_features.to_dict(orient= 'records'))
    result = clf.predict(test_features)

    #导出数据
    test_data['Survived'] = result
    test_data.to_csv('result.csv') 

    #准确率
    acc_decision_tree = round(clf.score(train_features, train_labels),6)
    print(acc_decision_tree)

    #提取出存活结果的列表
    df = pd.read_csv('test.csv')
    survived_correct = np.array(df['Survived']).tolist()
    print(survived_correct)

    #提取出存活结果的列表
    df = pd.read_csv('result.csv')
    survived_result = np.array(df['Survived']).tolist()
    print(survived_result)

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