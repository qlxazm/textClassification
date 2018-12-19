# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
import time
import math
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from textClassifier.utils import loadDataSet

class MutipleNativBayes(object):

    def __init__(self, alpha = 1):
        self.prioriProbability = None       #先验概率(取对数后)
        self.conditionalProbability = None  #条件概率(取对数后)
        self.wordBag = None                 #词袋
        self.__alpha = alpha                #平滑指数（默认为1）
        self.vectorSize = None              #样本向量大小
        self.__classNum = None              #分类样本数
        self.__numOfClass = None            #每个类的文本数
        self.__encoder = preprocessing.LabelEncoder()  #编码器
        self.__classIndex = None             #类别整数编号与索引的参照表

    def __selectFeature(self, texts, labels, max_features):
        """
        提取特征词袋并生成文本对应的特征向量
        :param texts:  文本集合
        :param label:  文本类别
        :param max_features: 提取的特征词袋的大小，默认是10000
        :return: 文本集合对应的文本向量
        """
        vectorizer = CountVectorizer(np.uint8)
        texts = vectorizer.fit_transform(texts)

        # 使用TF-IDF进行加权
        transformer = TfidfTransformer()
        texts = transformer.fit_transform(texts)

        #利用卡方检验提取特征值
        label = labels.astype(np.uint8)
        selector = SelectKBest(chi2, k=max_features)
        selectedFeatures = selector.fit_transform(texts, label)
        # 将单词与索引的对应关系转换成 索引与单词的对应关系，以便于获取选择的特征值
        wordIndex = vectorizer.vocabulary_
        indexWord = {value: key for key, value in wordIndex.items()}
        self.wordBag = []
        index = 0
        for support in selector.get_support():
            if support:
                self.wordBag.append(indexWord[index])
            index += 1
        return selectedFeatures

    def __computePrioriProbability(self, labels):
        """
        计算先验概率
        :param labels:
        :return: 无
        """
        self.__numOfClass = pd.value_counts(labels)  # 各个类别文本的计数
        self.__classNum = len(self.__numOfClass)     # 文本的类别总数
        textNumber = len(labels)                     # 文本总数

        self.prioriProbability = []
        self.__classIndex = []
        k = labels[0]
        kAlpha = self.__alpha * self.__classNum
        for label in labels:
            if k != label:
                self.__classIndex.append(k)
                temp = (self.__numOfClass[k] + self.__alpha) / (textNumber + kAlpha)
                self.prioriProbability.append(math.log(temp))
                k = label
        temp = (self.__numOfClass[labels[-1]]) / (textNumber)
        self.prioriProbability.append(math.log(temp))
        self.__classIndex.append(labels[-1])

    def __createClassMatrix(self, labels):
        """
        创建类矩阵
        :param labels:
        :return: 类矩阵
        """
        row = []
        col = []
        data = []
        textCount = len(labels)
        k = 0
        another = False
        currClass = labels[0]
        for i in range(0, self.__classNum):
            for j in range(0, textCount):
                row.append(i)
                col.append(j)
                currClass = labels[k]
                if j < k or another:
                    data.append(0)
                else:
                    k += 1
                    data.append(1)
                if k < textCount and labels[k] != currClass:
                    another = True
            another = False
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        return csr_matrix((data, (row, col)), shape=(self.__classNum, textCount))

    def __createTextVector(self, testDataset=None):
        """
        根据词袋生成测试集合的矩阵表示
        :param testDataset:
        :return:
        """
        X = CountVectorizer(vocabulary = self.wordBag, dtype=np.uint8).fit_transform(testDataset)
        transformer = TfidfTransformer()
        texts = transformer.fit_transform(X)
        return texts

    def  fit(self, texts, labels, max_features =10000):
        """
        训练
        :param texts:
        :param labels:
        :param max_features:
        :return:
        """
        # 编码类别
        labels = self.__encoder.fit_transform(labels)

        self.vectorSize = max_features                                  #设置文本向量的大小
        self.__computePrioriProbability(labels)                         #计算先验概率
        textMatrix = self.__selectFeature(texts, labels, max_features)  #提取特征值并获取训练集合的矩阵表示
        classMatrix = self.__createClassMatrix(labels)                  #创建类矩阵

        """
        使用矩阵求解先验概率
        """
        self.conditionalProbability = (classMatrix * textMatrix).toarray()
        self.conditionalProbability = self.conditionalProbability + self.__alpha #加上平滑指数
        tempMatrix = self.conditionalProbability.sum(axis=1)                     #按行相加求总和
        self.conditionalProbability = self.conditionalProbability / tempMatrix[:,None]
        self.conditionalProbability = np.log(self.conditionalProbability)        #取对数

    def predict(self, testDataset, labels):
        """
        预测
        :param testDataset:
        :param labels:
        :return:
        """
        # 编码类别
        labels = self.__encoder.fit_transform(labels)
        #将预测集合表示成矩阵形式
        testMatrix = self.__createTextVector(testDataset)
        testNum = len(labels)

        correct = 0
        incorrect = 0
        predictLable = np.zeros((testNum,1))
        predictProb = testMatrix * self.conditionalProbability.T + np.array(self.prioriProbability)
        for i in range(0,testNum):
            index = np.argmax(predictProb[i])
            predictLable[i][0] = self.__classIndex[index]
            if predictLable[i][0] == labels[i]:
                correct += 1
            else:
                incorrect += 1
        print("正确率是：{0}".format(correct / testNum))


if __name__ == '__main__':
    ticks = time.time()
    TRAIN_TABLE_NAME = "traindataset"   #训练集合所在的数据库表
    TEST_TABLE_NAME = "testdataset"     #测试集合所在的数据库表
    MAX_FEATURES = 100000               #选取的特征数特征数

    # 1、加载训练集
    X, y = loadDataSet(TRAIN_TABLE_NAME)
    mutipleNativBayes =MutipleNativBayes()
    mutipleNativBayes.fit(X, y, max_features = MAX_FEATURES)
    # 2、加载测试集
    X, y = loadDataSet(TEST_TABLE_NAME)
    mutipleNativBayes.predict(X, y)

    print("用时：{0}".format((time.time()-ticks) / 60))


