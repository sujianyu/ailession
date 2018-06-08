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
#根据在exploreBikeshare.py程序中数据探索后，符合正态分布，适用于线性回归

#显示权重的特征

#将数据分割训练数据与测试数据
#将数据分为2011年和2012年两组数据
_2011data = daydata.loc[daydata.yr == 0]
_2012data = daydata.loc[daydata.yr == 1]
X_train = _2011data.drop(['cnt', 'dteday', 'yr'], axis=1)
y_train = _2011data["cnt"]

X_test = _2012data.drop(['cnt', 'dteday', 'yr'], axis=1)
y_test = _2012data["cnt"]
print(X_train.shape)
print(y_train.shape)
columns = X_test.columns
# 数据标准化
from sklearn.preprocessing import StandardScaler
# 分别初始化对特征和目标值的标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

#对y做标准化--可以比较不做标准化的差异

y_train = ss_y.fit_transform(y_train.values.reshape(-1, 1))
y_test = ss_y.transform(y_test.values.reshape(-1, 1))

# 线性回归
from sklearn.linear_model import LinearRegression

# 使用默认配置初始化
lr = LinearRegression()
# 训练模型参数
lr.fit(X_train, y_train)

# 预测
y_test_pred_lr = lr.predict(X_test)
y_train_pred_lr = lr.predict(X_train)

print(lr.coef_.T)


# 看看各特征的权重系数，系数的绝对值大小可视为该特征的重要性
fs = pd.DataFrame({"columns": list(columns), "coef": list((lr.coef_.T))})
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

#设置超参数（正则参数）范围
alphas = [ 0.01, 0.1, 1, 10,100]

#生成一个RidgeCV实例
ridge = RidgeCV(alphas=alphas, store_cv_values=True)

#模型训练
ridge.fit(X_train, y_train)

#预测
y_test_pred_ridge = ridge.predict(X_test)
y_train_pred_ridge = ridge.predict(X_train)


# 评估，使用r2_score评价模型在测试集和训练集上的性能
print ('The r2 score of RidgeCV on test is', r2_score(y_test, y_test_pred_ridge))
print ('The r2 score of RidgeCV on train is', r2_score(y_train, y_train_pred_ridge))

mse_mean = np.mean(ridge.cv_values_, axis = 0)
plt.plot(np.log10(alphas), mse_mean.reshape(len(alphas),1))

#这是为了标出最佳参数的位置，不是必须
#plt.plot(np.log10(ridge.alpha_)*np.ones(3), [0.28, 0.29, 0.30])

plt.xlabel('log(alpha)')
plt.ylabel('mse')
plt.show()

print ('alpha is:', ridge.alpha_)

# 看看各特征的权重系数，系数的绝对值大小可视为该特征的重要性
fs = pd.DataFrame({"columns":list(columns), "coef_lr":list((lr.coef_.T)), "coef_ridge":list((ridge.coef_.T))})
fs.sort_values(by=['coef_lr'],ascending=False)

# Lasso／L1正则
from sklearn.linear_model import LassoCV

#设置超参数搜索范围
#alphas = [ 0.01, 0.1, 1, 10,100]

#生成一个LassoCV实例
#lasso = LassoCV(alphas=alphas)
lasso = LassoCV()

#训练（内含CV）
lasso.fit(X_train, y_train)

#测试
y_test_pred_lasso = lasso.predict(X_test)
y_train_pred_lasso = lasso.predict(X_train)


# 评估，使用r2_score评价模型在测试集和训练集上的性能
print ('The r2 score of LassoCV on test is', r2_score(y_test, y_test_pred_lasso))
print ('The r2 score of LassoCV on train is', r2_score(y_train, y_train_pred_lasso))
mses = np.mean(lasso.mse_path_, axis=1)
plt.plot(np.log10(lasso.alphas_), mses)
# plt.plot(np.log10(lasso.alphas_)*np.ones(3), [0.3, 0.4, 1.0])
plt.xlabel('log(alpha)')
plt.ylabel('mse')
plt.show()


print ('alpha is:', lasso.alpha_)

# 看看各特征的权重系数，系数的绝对值大小可视为该特征的重要性
fs = pd.DataFrame({"columns": list(columns), "coef_lr": list((lr.coef_.T)), "coef_ridge": list((ridge.coef_.T)),
                   "coef_lasso": list((lasso.coef_.T))})
fs.sort_values(by=['coef_lr'], ascending=False)

mses = np.mean(lasso.mse_path_, axis=1)
plt.plot(np.log10(lasso.alphas_), mses)
# plt.plot(np.log10(lasso.alphas_)*np.ones(3), [0.3, 0.4, 1.0])
plt.xlabel('log(alpha)')
plt.ylabel('mse')
plt.show()

print ('alpha is:', lasso.alpha_)

