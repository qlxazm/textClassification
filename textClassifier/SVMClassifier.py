# -*- coding: utf-8 -*-
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from SQLHelper.SqlHelper import SqlHelper
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from textClassifier.utils import loadDataSet
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import time

class SVMClassifier(object):

    def __init__(self):
        self.wordBag = None     #词袋
        self.__trainX = None    #训练数据集
        self.__trainY = None    #训练数据集label
        self.__classNum = None  # 分类样本数
        self.__encoder = preprocessing.LabelEncoder()  # 编码器
        self.__SVM = LinearSVC(random_state=0, tol=1e-5, class_weight="balanced")
        # self.__SVM = SGDClassifier(tol=0.01, n_jobs=-1, shuffle=True, average=True)      #分类器  使用两个CPU核  重排训练集合  随机梯度下降算法

    def __selectFeature(self, X,  labels, maxFeature=10000):
        """
        使用TF-IDF加权之后使用卡方检验选择特征
        :param X:
        :param labels:
        :param maxFeature:
        :return:
        """
        vectorizer = CountVectorizer(np.uint8)
        texts = vectorizer.fit_transform(X)

        #使用TF-IDF进行加权
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(texts)

        # 利用卡方检验提取特征值
        labels = labels.astype(np.uint8)
        selector = SelectKBest(chi2, k=maxFeature)
        self.__trainX = selector.fit_transform(tfidf, labels)
        # 正则化
        self.__trainX = preprocessing.normalize(self.__trainX, norm='l2')

        # 将单词与索引的对应关系转换成 索引与单词的对应关系，以便于获取选择的特征值
        wordIndex = vectorizer.vocabulary_
        indexWord = {value: key for key, value in wordIndex.items()}
        self.wordBag = []
        index = 0
        for support in selector.get_support():
            if support:
                self.wordBag.append(indexWord[index])
            index += 1

    def __createTestMatrix(self, X=None):
        """
        生成测试数据集的矩阵表示
        :param X:
        :return:
        """
        if X == None:
            return None
        X = CountVectorizer(vocabulary=self.wordBag, dtype=np.uint8).fit_transform(X)
        # 使用TF-IDF进行加权
        transformer = TfidfTransformer()
        X = transformer.fit_transform(X)
        # 正则化
        return preprocessing.normalize(X, norm='l2')

    def fitTransform(self, X, labels, maxFeature=30000):
        """
        训练模型并持久化
        :param X:
        :param labels:
        :param maxFeature:
        :return:
        """
        # 编码类别
        self.__trainY = self.__encoder.fit_transform(labels)
        try:
            self.__SVM = joblib.load("SVM.m")
        except FileNotFoundError:
            # 打乱数据
            X, labels = shuffle(X,labels, random_state=0)
            #特征选择
            self.__selectFeature(X, self.__trainY, maxFeature)
            # self.__SVM.fit(self.__trainX, self.__trainY)
            # 分类器参数调优
            svm_clf = self.__SVM
            param_grid = [
                {
                    'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
                 }
            ]
            gridSearch = GridSearchCV(svm_clf, param_grid, scoring='accuracy', cv=5)
            gridSearch.fit(self.__trainX, self.__trainY)
            self.__SVM = gridSearch.best_estimator_
            # 保存模型到本地文件
            joblib.dump(self.__SVM, "SVM.m")


    def predict(self, X, labels):
        """
        预测
        :param X:
        :param labels:
        :return:
        """
        # 编码类别
        labels = self.__encoder.transform(labels)
        X = self.__createTestMatrix(X)
        preLables = self.__SVM.predict(X)
        print("正确率：%f" % (np.mean(preLables == labels))) #准确度
        print(confusion_matrix(labels, preLables) )          #混淆矩阵


if __name__ == '__main__':
    ticks = time.time()

    TRAIN_TABLE_NAME = "traindataset"  # 训练集合所在的数据库表
    TEST_TABLE_NAME = "testdataset"    # 测试集合所在的数据库表
    MAX_FEATURE = 30000                #选取的特征数

    # 1、加载训练集
    X, y = loadDataSet(tableName = TRAIN_TABLE_NAME)
    svmClassifier = SVMClassifier()
    svmClassifier.fitTransform(X, y, maxFeature = MAX_FEATURE)

    # 2、加载测试集
    X, y = loadDataSet(tableName = TEST_TABLE_NAME)
    svmClassifier.predict(X, y)

    print("用时：{0}分钟".format((time.time() - ticks) / 60))


