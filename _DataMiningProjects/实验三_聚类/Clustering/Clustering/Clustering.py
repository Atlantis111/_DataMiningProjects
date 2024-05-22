import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
 
 
#计算两点间的距离
def get_distance(p1, p2):
    diff = [x-y for x, y in zip(p1, p2)]
    distance = np.sqrt(sum(map(lambda x: x**2, diff)))
    return distance
 
 
# 计算多个点的中心，输入参数为列表，输出中心点
def getnew_center_point(cluster):
    N = len(cluster)
    m = np.matrix(cluster).transpose().tolist()
    center_point = [sum(x)/N for x in m]
    return center_point
 
 
# 检查旧、新中心点是否有差别
def check_center_diff(center, new_center):
    n = len(center)
    for c, nc in zip(center, new_center):
        if c != nc:
            return False
    return True
 
 
# K-means算法的实现
def K_means(samples, center_points):
 
    N = len(samples)         # 样本个数
    n = len(samples[0])      # 单个样本的属性个数10
    k = len(center_points)  # 类别数3
 
    tot = 0
    while True:             # 迭代
        temp_center_points = [] # 记录中心点
        clusters = []       # 记录聚类的结果
        for c in range(0, k):
            clusters.append([]) # 初始化，此时clusters=[[],[],[]]
 
        # 针对每个点，寻找距离其最近的中心点（寻找组织）
        for i, data in enumerate(samples):
            distances = []
            for center_point in center_points:
                distances.append(get_distance(data, center_point))
            index = distances.index(min(distances)) # 找到最小的距离的那个中心点的索引，
 
            clusters[index].append(data)    # 那么这个中心点代表的簇，里面增加一个样本.
 
        tot += 1
        print(tot, '次迭代', clusters)
        k = len(clusters) 

        colors = ['r.', 'g.', 'b.'] 
        for i, cluster in enumerate(clusters):
            data = np.array(cluster)     
            data_x = [x[8] for x in data]
            data_y = [x[9] for x in data]
            plt.subplot(4, 3, tot)
            plt.plot(data_x, data_y, colors[i])
            plt.axis([0, 300, 0, 200])
 
        # 计算新的中心点
        for cluster in clusters:
            temp_center_points.append(getnew_center_point(cluster))
 
        # 在计算中心点的时候，需要将原来的中心点算进去
        for j in range(0, k):
            if len(clusters[j]) == 0:
                temp_center_points[j] = center_points[j]
 
        # 判断中心点是否发生变化
        for c, nc in zip(center_points, temp_center_points):
            if not check_center_diff(c, nc):
                center_points = temp_center_points[:]   # 复制一份
                break
        else: 
            break
 
    plt.show()
    return clusters # 返回聚类的结果
 
 
 
 
# 随机获取一个样本集，用于测试K-means算法
def get_test_data():
 
    samples = []
    center_points = []

    df = pd.read_csv('hcvdat.csv')

    df.ALB = df.ALB.fillna(df.ALB.mean())
    df.ALP = df.ALP.fillna(df.ALP.mean())
    df.ALT = df.ALT.fillna(df.ALT.mean())
    df.CHOL = df.CHOL.fillna(df.CHOL.mean())
    df.PROT = df.PROT.fillna(df.PROT.mean())

    df.drop(['Number','Category','Age','Sex'], axis = 1, inplace = True)
    samples = df.values.tolist()          #提取属性列表

    center_points = [[38.5,52.5,7.7,22.1,7.5,6.93,3.23,106,12.1,69], 
                     [47,19.1,38.9,164.2,17,7.09,3.2,79.3,90.4,70.1], 
                     [41,43.1,2.4,83.5,6,11.49,5.42,55.2,130,66.5]]
 
    return samples, center_points
 
 
if __name__ == '__main__':
    
    TP_1,FP_1,FN_1=0,0,0
    TP_2,FP_2,FN_2=0,0,0
    TP_3,FP_3,FN_3=0,0,0

    samples, center_points = get_test_data()
    clusters = K_means(samples, center_points)

    print('分类结果')
    print('\n')
    for i, cluster in enumerate(clusters):
        print('cluster ', i, ' ', cluster)
        print('\n')

    for n, point in enumerate(samples):
        for m, cluster in enumerate(clusters):
            if point in cluster:
                print(n+1,'条数据在第',m+1,'类中')

                if m+1==1 and n+1<=540:   TP_1=TP_1+1     #预测为1，实际也为1
                if m+1==1 and n+1>540:   FP_1=FP_1+1     #预测为1，实际不为1
                if m+1!=1 and n+1<=540:    FN_1=FN_1+1     #预测不为1，实际为1

                if m+1==2 and n+1>540 and n+1<=585:   TP_2=TP_2+1     #预测为2，实际也为2
                if m+1==2 and (n+1<=540 or n+1>585):   FP_2=FP_2+1     #预测为2，实际不为2
                if m+1!=2 and n+1>540 and n+1<=585:    FN_2=FN_2+1     #预测不为2，实际为2
                
                if m+1==3 and n+1>=586:   TP_3=TP_3+1     #预测为3，实际也为3
                if m+1==3 and n+1<586:   FP_3=FP_3+1     #预测为3，实际不为3
                if m+1!=3 and n+1>=586:    FN_3=FN_3+1     #预测不为3，实际为3

    print('TP_1=',TP_1)
    print('FP_1=',FP_1)
    print('FN_1=',FN_1)
    print('TP_2=',TP_2)
    print('FP_2=',FP_2)
    print('FN_2=',FN_2)
    print('TP_3=',TP_3)
    print('FP_3=',FP_3)
    print('FN_3=',FN_3)