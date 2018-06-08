#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score  #评价回归预测模型的性能
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

#对y做标准化--可以比较不做标准化的差异
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

# 线性回归
from sklearn.linear_model import LinearRegression

# 使用默认配置初始化
lr = LinearRegression()
# 训练模型参数
lr.fit(X_train, y_train)

# 预测
y_test_pred_lr = lr.predict(X_test)
y_train_pred_lr = lr.predict(X_train)

# 看看各特征的权重系数，系数的绝对值大小可视为该特征的重要性
fs = pd.DataFrame({"columns":list(columns), "coef": list((lr.coef_.T))})
fs.sort_values(by=['coef'], ascending=False)

# 使用r2_score评价模型在测试集和训练集上的性能，并输出评估结果
#测试集
print('The r2 score of LinearRegression on test is', r2_score(y_test, y_test_pred_lr))
#训练集
print('The r2 score of LinearRegression on train is', r2_score(y_train, y_train_pred_lr))
#在训练集上观察预测残差的分布，看是否符合模型假设：噪声为0均值的高斯噪声
f, ax = plt.subplots(figsize=(7, 5))
f.tight_layout()
ax.hist(y_train - y_train_pred_lr,bins=40, label='Residuals Linear', color='b', alpha=.5)
ax.set_title("Histogram of Residuals")
ax.legend(loc='best')

plt.figure(figsize=(4, 3))
plt.scatter(y_train, y_train_pred_lr)
plt.plot([-3, 3], [-3, 3], '--k')   #数据已经标准化，3倍标准差即可
plt.axis('tight')
plt.xlabel('True price')
plt.ylabel('Predicted price')
plt.tight_layout()

#岭回归／L2正则
from sklearn.linear_model import  RidgeCV

