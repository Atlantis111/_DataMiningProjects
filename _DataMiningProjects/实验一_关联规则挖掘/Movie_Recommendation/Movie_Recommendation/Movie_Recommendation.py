from __future__ import print_function
import pandas as pd


#Apriori算法：

#连接函数，用于实现频繁项集到候选项集的连接
def connect_string(x, ms):
  x = list(map(lambda i:sorted(i.split(ms)), x))
  l = len(x[0])
  r = []
  for i in range(len(x)):
    for j in range(i,len(x)):
      if x[i][:l-1] == x[j][:l-1] and x[i][l-1] != x[j][l-1]:
        r.append(x[i][:l-1]+sorted([x[j][l-1],x[i][l-1]]))
  return r


#寻找关联规则的函数，放入参数分别为初始候选项数据，最小支持度，最小置信度，候选项元素的连接符
def find_rule(d, support, confidence, ms = u'--'):
  result = pd.DataFrame(index=['support', 'confidence'])    #定义输出结果

  support_series = 1.0*d.sum()/len(d)   #定义支持度序列
  column = list(support_series[support_series > support].index)     #用输入的支持度筛选并输出
  k = 0

  #column为项集元素的长度
  while len(column) > 1:
    k = k+1
    print(u'\n正在进行第%s次搜索...' %k)    #打印搜索过程
    column = connect_string(column, ms)     #建立连接
    print(u'数目：%s...' %len(column))
    sf = lambda i: d[i].prod(axis=1, numeric_only = True)   #新项集支持度的计算

    #创建连接数据
    d_2 = pd.DataFrame(list(map(sf,column)), index = [ms.join(i) for i in column]).T

    support_series_2 = 1.0*d_2[[ms.join(i) for i in column]].sum()/len(d)       #计算新项集的支持度
    column = list(support_series_2[support_series_2 > support].index)       #筛选新项集，得出其中的频繁项集
    support_series = support_series.append(support_series_2)
    column2 = []

    #对最后的频繁项集，计算其置信度，遍历所有顺序，{A,B,C}是A+B-->C，B+C-->A还是C+A-->B？
    for i in column: 
      i = i.split(ms)
      for j in range(len(i)):
        column2.append(i[:j]+i[j+1:]+i[j:j+1])

    cofidence_series = pd.Series(index=[ms.join(i) for i in column2],dtype=float) #定义置信度序列，join()方法用于将序列中的元素以指定的字符连接生成一个新的字符串
                                                                      #将column2里的元素用连接符连接,由于可能出现空序列，所以要将dtype显式的说明为float类型

    for i in column2:       #计算置信度序列
      cofidence_series[ms.join(i)] = support_series[ms.join(sorted(i))]/support_series[ms.join(i[:len(i)-1])]   #sorted()对所有可迭代的对象进行排序操作

    for i in cofidence_series[cofidence_series > confidence].index:     #根据输入的最小置信度对最后一组频繁项集进行筛选
      result[i] = 0.0
      result[i]['confidence'] = cofidence_series[i]
      result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]

  result = result.T.sort_values(['confidence','support'], ascending = False)    #结果输出
  print(u'\n结果为：')
  print(result)
  return result

id

inputfile = '../../SortOutMovieData/SortOutMovieData/Movie.xlsx'     #导入文件
outputfile = 'Movie_Recommendation.xlsx'    #结果文件
data = pd.read_excel(inputfile, header = None)          #用pd.read_excel()函数读入数据，header=None表示没有列名

print(u'\n转换原始数据至0-1矩阵...')
ct = lambda x : pd.Series(1, index = x[pd.notnull(x)])      #转换0-1矩阵的过渡函数，pd.Series()函数即创建一维数组，notnull函数将值返回成bool型数组
b = map(ct, data.values)                #用map方式执行,as_matrix()在新版的pandas中已被更新，换成了values(values是属性不是方法，不添加括号)，map()即返回多个值在函数计算后得到的映射
data = pd.DataFrame(list(b)).fillna(0)          #矩阵转换，空值用0填充
print(u'\n转换完毕。')
del b

support = 0.1       #输入最小支持度
confidence = 0.1    #输入最小置信度
ms = '---'          #项集内各元素的连接符，默认'--'，需要保证原始表格中不含有该字符

find_rule(data, support, confidence, ms).to_excel(outputfile)       #保存结果至结果文件夹中
