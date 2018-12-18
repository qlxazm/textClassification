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
from sklearn.externals import joblib
import time

class SVMClassifier(object):

    def __init__(self):
        self.wordBag = None     #词袋
        self.__trainX = None    #训练数据集
        self.__trainY = None    #训练数据集label
        self.__classNum = None  # 分类样本数
        self.__encoder = preprocessing.LabelEncoder()  # 编码器
        self.__SVM = SGDClassifier(tol=0.01, n_jobs=-1, shuffle=True, average=True)      #分类器  使用两个CPU核  重排训练集合  随机梯度下降算法

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
        # labels = labels.astype(np.uint8)
        # selector = SelectKBest(chi2, k=maxFeature)
        # self.__trainX = selector.fit_transform(tfidf, labels)
        # 正则化
        # self.__trainX = preprocessing.normalize(self.__trainX, norm='l2')
        self.__trainX = preprocessing.normalize(tfidf, norm='l2')

        # 将单词与索引的对应关系转换成 索引与单词的对应关系，以便于获取选择的特征值
        # wordIndex = vectorizer.vocabulary_
        # indexWord = {value: key for key, value in wordIndex.items()}
        # self.wordBag = []
        # index = 0
        # for support in selector.get_support():
        #     if support:
        #         self.wordBag.append(indexWord[index])
        #     index += 1

    def __createTestMatrix(self, X=None):
        """
        生成测试数据集的矩阵表示
        :param X:
        :return:
        """
        if X == None:
            return None
        # X = CountVectorizer(vocabulary=self.wordBag, dtype=np.uint8).fit_transform(X)
        X = CountVectorizer(dtype=np.uint8).fit_transform(X)
        # 使用TF-IDF进行加权
        transformer = TfidfTransformer()
        X = transformer.fit_transform(X)
        # 正则化
        return preprocessing.normalize(X, norm='l2')

    def fitTransform(self, X, labels, maxFeature=10000):
        # 编码类别
        self.__trainY = self.__encoder.fit_transform(labels)
        self.__selectFeature(X, self.__trainY, maxFeature)

        # 分类器参数调优
        svm_clf = self.__SVM
        param_grid = [
            {'alpha': [0.0001, 0.001, 0.01, 0.1],
             'penalty': ['l1', 'l2'],
             }
        ]
        gridSearch = GridSearchCV(svm_clf, param_grid, scoring='accuracy', cv=10)
        gridSearch.fit(self.__trainX, self.__trainY)

        # 交叉验证后提取最佳参数
        # bestAlpha = gridSearch.best_params_["alpha"]
        # bestPenalty = gridSearch.best_params_["penalty"]
        #交叉验证完使用最佳的SVM
        self.__SVM = gridSearch.best_estimator_

        #保存模型
        joblib.dump(self.__SVM, "SVM.m")

    def predict(self, X, labels):
        # 编码类别
        labels = self.__encoder.transform(labels)

        X = self.__createTestMatrix(X)
        preLables = self.__SVM.predict(X)

        print("正确率：%f" % ( np.mean(preLables == labels)))

if __name__ == '__main__':
    ticks = time.time()
    TRAIN_TABLE_NAME = "traindataset"  # 训练集合所在的数据库表
    TEST_TABLE_NAME = "testdataset"    # 测试集合所在的数据库表
    MAX_FEATURE = 30000
    conditions = ["id >= 1"]

    encoder = preprocessing.LabelEncoder()

    # 1、加载训练集
    train_text = SqlHelper().commonSelect(tableName=TRAIN_TABLE_NAME, params=["content"], conditions=conditions)
    train_text = [raw[0].decode() for raw in train_text]
    train_label = SqlHelper().commonSelect(tableName=TRAIN_TABLE_NAME, params=["type"], conditions=conditions)
    train_label = [raw[0] for raw in train_label]

    svmClassifier = SVMClassifier()
    svmClassifier.fitTransform(train_text, train_label, MAX_FEATURE)

    # 2、加载测试集
    test_text = SqlHelper().commonSelect(tableName=TEST_TABLE_NAME, params=["content"], conditions=conditions)
    test_text = [raw[0].decode() for raw in test_text]
    test_label = SqlHelper().commonSelect(tableName=TEST_TABLE_NAME, params=["type"], conditions=conditions)
    test_label = [raw[0] for raw in test_label]

    test_text, test_label = shuffle(test_text, test_label)

    svmClassifier.predict(test_text[10000], test_label[10000])

    print("用时：%f" % ((time.time() - ticks) / 60))


