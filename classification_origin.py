import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
from sklearn.tree import DecisionTreeClassifier


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report

# 步骤1: 数据准备
# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data  # 特征数据
y = iris.target  # 目标标签

# 只选择山鸢尾（类别0）和变色鸢尾（类别1）的样本
X = X[(y == 0) | (y == 1)]
y = y[(y == 0) | (y == 1)]

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤2: 模型训练
# 创建SVM分类器
clf = svm.SVC(kernel='linear', C=1)

# 训练模型
clf.fit(X_train, y_train)

# 步骤3: 模型评估
# 使用测试集进行预测
y_pred = clf.predict(X_test)

# 计算准确度
accuracy = accuracy_score(y_test, y_pred)
print("准确度：", accuracy)

# 打印分类报告，包括精确度、召回率和F1值等
print(classification_report(y_test, y_pred))





# import the iris data set
iris = pd.read_csv('IRIS.csv')
iris = iris.dropna()
# print(iris.head())


# analyze the data
# print(iris.describe())

report = ProfileReport(iris)
# report.to_file('report.html')

"""By the report.html, we find that the length and width of petal have more correlation with species than sepal,
those, we want to divide the training sets and test sets into three type(all features, only petal, only sepal), and set up a
comparison test
"""
# all features
# divide features and target
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris['species']

"""
# encode species as 0,1,2
encode = LabelEncoder()
y = encode.fit_transform(y)
# print(y)
"""

# divide data set into train sets and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

"""
print('X train shape: {}'.format(X_train.shape))
print('X test shape: {}'.format(X_test.shape))
print('y train shape: {}'.format(y_train.shape))
print('y train shape: {}'.format(y_test.shape))
"""

# only petal
petal = iris[['petal_length', 'petal_width', 'species']]
train_p, test_p = train_test_split(petal, test_size=0.3, random_state=0)
X_train_p = train_p[['petal_length', 'petal_width']]
y_train_p = train_p.species
X_test_p = test_p[['petal_length', 'petal_width']]
y_test_p = test_p.species

"""
print('X_p train shape: {}'.format(X_train_p.shape))
print('X_p test shape: {}'.format(X_test_p.shape))
print('y_p train shape: {}'.format(y_train_p.shape))
print('y_p train shape: {}'.format(y_test_p.shape))
"""

# only sepal
sepal = iris[['sepal_length', 'sepal_width', 'species']]
train_s, test_s = train_test_split(sepal, test_size=0.2, random_state=10)
X_train_s = train_s[['sepal_length', 'sepal_width']]
y_train_s = train_s.species
X_test_s = test_s[['sepal_length', 'sepal_width']]
y_test_s = test_s.species

"""
print('X_s train shape: {}'.format(X_train_s.shape))
print('X_s test shape: {}'.format(X_test_s.shape))
print('y_s train shape: {}'.format(y_train_s.shape))
print('y_s train shape: {}'.format(y_test_s.shape))
"""

# train and predict


"""based on Decision Tree Classifier"""
print('*****Based on Decision Tree Classifier*****')
# for all features
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_test_pre = dt.predict(X_test)
total = np.count_nonzero(y_test)
c = np.count_nonzero(y_test_pre == y_test)
print('right prediction number：{}/{}'.format(c, total))
print('The accuracy of the Decision Tree using all features is: {:.3f}\n'.format(metrics.accuracy_score(y_test_pre, y_test)))

# for petal
dt = DecisionTreeClassifier()
dt.fit(X_train_p, y_train_p)
y_test_pre_p = dt.predict(X_test_p)
total = np.count_nonzero(y_test_p)
c = np.count_nonzero(y_test_pre_p == y_test_p)
print('right prediction number：{}/{}'.format(c, total))
print('The accuracy of the Decision Tree using Petal is: {:.3f}\n'.format(metrics.accuracy_score(y_test_pre_p, y_test_p)))

# for sepal
dt = DecisionTreeClassifier()
dt.fit(X_train_s, y_train_s)
y_test_pre_s = dt.predict(X_test_s)
total = np.count_nonzero(y_test_s)
c = np.count_nonzero(y_test_pre_s == y_test_s)
print('right prediction number：{}/{}'.format(c, total))
print('The accuracy of the Decision Tree using sepal is: {:.3f}\n'.format(metrics.accuracy_score(y_test_pre_s, y_test_s)))

"""based on Random Forest Classifier"""
print('*****Based on Random Forest Classifier*****')
# for all features test set
rf = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=10)
rf.fit(X_train, y_train)
y_test_pre_rf = rf.predict(X_test)
accuracy = rf.score(X_test, y_test)
total = np.count_nonzero(y_test)
c = np.count_nonzero(y_test_pre_rf == y_test)
print('Test Set Prediction')
print('right prediction number：{}/{}'.format(c, total))
print('Accuracy:', accuracy)
print("\n")

# for all features training set
y_train_pre_rf = rf.predict(X_train)
accuracy_tr = rf.score(X_train, y_train)
total_tr = np.count_nonzero(y_train)
c_tr = np.count_nonzero(y_train_pre_rf == y_train)
print('Training Set Prediction')
print('right prediction number：{}/{}'.format(c_tr, total_tr))
print('Accuracy:', accuracy_tr)
