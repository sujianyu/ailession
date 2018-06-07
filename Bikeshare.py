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
#读取数据
dbpath = "f:\\aidata"
daydata = pd.read_csv(dbpath + "\day.csv")
print(daydata.head())
print(daydata.shape)
#根据在exploreBikeshare.py程序中数据探索后，符合正态分布，适用于线性回归

#显示权重的特征
columns = daydata.columns
#将数据分割训练数据与测试数据
#将数据分为2011年和2012年两组数据
_2011data = daydata.loc[daydata.yr == 0]
_2012data = daydata.loc[daydata.yr == 1]
X_train = _2011data.drop("cnt", axis=1)
y_train = _2011data["cnt"]

X_test = _2012data.drop("cnt", axis=1)
y_test = _2012data["cnt"]
print(X_train.shape)
print(X_test.shape)

# 数据标准化
from sklearn.preprocessing import StandardScaler
# 分别初始化对特征和目标值的标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
