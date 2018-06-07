#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#解决中文显示问题
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from pylab import *
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']#指定默认字体
mpl.rcParams['axes.unicode_minus'] =False # 解决保存图像是负号'-'显示为方块的问题

dbpath = "f:\\aidata"
daydata = pd.read_csv(dbpath + "\day.csv")
#读入day.csv文件的数据
print(daydata.describe())
#将数据分为2011年和2012年两组数据
_2011data = daydata.loc[daydata.yr == 0]
_2012data = daydata.loc[daydata.yr == 1]
flag = plt.figure()
#正态分布图
sns.distplot(daydata.cnt, bins=30, kde=True)
plt.xlabel("共享单车数量",fontsize=12)
plt.show()
#从结果看，基本符合正态分布特征。
#单特征散点图
#数据分布情况
plt.scatter(range(daydata.shape[0]), daydata.cnt, color='purple')
plt.title("样本分布情况")
plt.show()
#各季节直方图
sns.barplot(x=daydata['season'], y=daydata["cnt"], data=daydata)
plt.xlabel("季节")
plt.ylabel("共享数量")
plt.show()

#两两特征之间相关性
cols = daydata.columns
#计算相关系数
data_corr = daydata.corr().abs()
plt.subplots(figsize=(13, 9))
sns.heatmap(data_corr, annot=True)
#plt.savefig("house_coor.png")
plt.show()
#将强相关的特征对提取出来
threshold = 0.5
corr_list = []
size = data_corr.shape[0]
for i in range(0,size):
    for j in range(i+1,size):
        corr = data_corr.iloc[i,j]
        if (corr>threshold and corr<1) or ( corr <0 and corr<=threshold):
            corr_list.append([corr, i, j])
s_corr_list = sorted(corr_list,key=lambda x:-abs(x[0]))
for v,i,j in s_corr_list:
    print("%s and %s =%.2f" % (cols[i], cols[j], v))

#显示相关系数图
for v,i,j in s_corr_list:
    sns.pairplot(daydata, size=6, x_vars=cols[i], y_vars=cols[j] )
    plt.show()



