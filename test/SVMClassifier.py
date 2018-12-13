# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from SQLHelper.SqlHelper import SqlHelper
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
import time

class SVMClassifier(object):

    def __init__(self):
        self.wordBag = None     #词袋
        self.__trainX = None    #训练数据集
        self.__trainY = None    #训练数据集label
        self.__classNum = None  # 分类样本数
        self.__encoder = preprocessing.LabelEncoder()  # 编码器
        self.__SVM = SVC()      #分类器

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
        return transformer.fit_transform(X)

    def fitTransform(self, X, labels, maxFeature=10000):
        # 编码类别
        self.__trainY = self.__encoder.fit_transform(labels)
        self.__selectFeature(X, self.__trainY, maxFeature)
        self.__SVM.fit(self.__trainX, self.__trainY)

    def predict(self, X, labels):
        # 编码类别
        labels = self.__encoder.transform(labels)
        X = self.__createTestMatrix(X)
        preLables = self.__SVM.predict(X)

        correct = 0
        for i in range(len(preLables)):
            if preLables[i] == labels[i]:
                correct += 1

        print("正确率：%f" % (correct / len(preLables)))

if __name__ == '__main__':
    ticks = time.time()

    TRAIN_TABLE_NAME = "traindataset"  # 训练集合所在的数据库表
    TEST_TABLE_NAME = "testdataset"    # 测试集合所在的数据库表
    MAX_FEATURE = 10000
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
    svmClassifier.predict(test_text, test_label)

    print("用时：%f" % ((time.time() - ticks) / 60))


