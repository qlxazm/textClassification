# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from SQLHelper.SqlHelper import SqlHelper
import time
import math
from scipy.sparse import csr_matrix


def createWordBag(texts, label, max_features=30000):
    """
    提取特征词袋并生成文本对应的特征向量
    :param texts:  文本集合
    :param label:  文本类别
    :param max_features: 提取的特征词袋的大小，默认是30000
    :return: 文本集合对应的文本向量  特征词袋
    """
    vectorizer = CountVectorizer(np.uint8)
    texts = vectorizer.fit_transform(texts)
    #将 单词与索引的对应关系 转换成 索引与单词的对应关系
    wordIndex = vectorizer.vocabulary_
    indexWord = {value: key for key, value in wordIndex.items()}

    label = label.astype(np.uint8)
    selector = SelectKBest(chi2, k=max_features)
    selectedFeatures = selector.fit_transform(texts, label)

    #获取选择的特征值
    words = []
    index = 0
    for support in selector.get_support():
        if support:
            words.append(indexWord[index])
        index += 1

    print(type(selectedFeatures))
    return selectedFeatures.toarray(), words

def train(textVector, label, alpha = 1):
    """
    根据训练集合计算先验概率和类别的概率
    :param textVector: 训练样本集合
    :param label:      样本的类别集合
    :param p:          m估计所使用的参数
    :return:(先验概率，类别概率)
    """
    #计算先验概率
    count = pd.value_counts(label)  #各个类别文本的计数
    print(len(count))
    classNumber = len(count)       #文本的类别总数
    textNumber = len(label)        #文本总数
    vectorLen = len(textVector[0]) #文本向量长度

    textProportion = []      #每类文本在整个文本集合中的比例
    for i in range(0,classNumber):
        print(count[i])
        temp = (count[i] + alpha) / (textNumber + classNumber * alpha)
        textProportion.append(math.log(temp))

    # #创建词频矩阵
    # row = []
    # col = []
    # data = []
    # for i in label:
    #     for j in range(0,vectorLen):
    #
    #
    #
    #
    # #先验概率组成的数组
    # shape = [classNumber, vectorLen, 2]
    # prioriProbability = np.zeros(shape,dtype=np.float32)
    # #计算条件概率
    # rowNumberTemp = 0
    # for i in range(0, classNumber):
    #     for j in range(0, vectorLen):
    #         temp = textVector[rowNumberTemp:(rowNumberTemp + count[i]),j:(j + 1)]
    #         #根据m估计计算先验概率p(j|i)，即已知在文本是i类的情况下，文本中包含j单词的条件概率
    #         noZero = countNoZero(temp)
    #         includeJ = (noZero + vectorLen * p) / (count[i] + vectorLen) #文本包含该单词的概率
    #         excludeJ = 1 - includeJ                                            #文本不包含该该单词的概率
    #         prioriProbability[i][j][0] = math.log(excludeJ)
    #         prioriProbability[i][j][1] = math.log(includeJ)
    #         rowNumberTemp = rowNumberTemp + count[i]
    #
    # return prioriProbability, textProportion

def createTextVector(text, wordBag):
    """
    根据文本text和词袋创建代表text的向量
    :param text:
    :param wordBag:
    :return:
    """
    vector = list()
    for word in wordBag:
        if word in text:
            vector.append(1)
        else:
            vector.append(0)
    return vector

def predictText(textVector,prioriProbability, textProportion, classNum, vectorLen):
    """
    返回最大值的索引
    :param textVector:
    :return:
    """
    max = 0
    maxIndex = 0
    for i in range(0,classNum):
        temp = -textProportion[i]
        temp -= prioriProbability[i][np.arange(vectorLen), textVector].sum()
        if temp > max:
            max = temp
            maxIndex = i
    return maxIndex

def predict(textDataSet, label, wordBag, prioriProbability, textProportion):
    #正确和错误的文本计数
    label = label.astype(np.uint8)
    correct = 0
    incorrect = 0
    index = 0
    classNum = len(textProportion)
    vectorLen = len(wordBag)

    start = time.time()
    thousend = 0
    for text in textDataSet:
        textVector = createTextVector(text, wordBag)
        predictClass = predictText(textVector, prioriProbability, textProportion, classNum, vectorLen)
        print("预测的类：%d" % predictClass)
        print(type(predictClass))
        print("真正的类：%d" % label[index])
        print(type(label[index]))
        if predictClass == label[index]: #预测正确
            correct += 1
        else:                            #预测失败
            incorrect += 1
        index += 1
        if index % 2000 == 0:
            thousend += 1
            print("测试了2000，用时：%f == 正确率 %f" % ((time.time() - start) / 60,correct / (thousend * 2000)))
            start =  time.time()
    print("正确率：")
    print(correct / len(textDataSet))
    return None

def countNoZero(arr):
    """
    计算arr中非零元素的个数
    :param arr:
    :return:
    """
    count = np.nonzero(arr)
    if len(count):
        return len(count[0])
    return 0



if __name__ == '__main__':
    ticks = time.time()
    TRAIN_TABLE_NAME = "traindataset"   #训练集合所在的数据库表
    TEST_TABLE_NAME = "testdataset"     #测试集合所在的数据库表
    conditions = ["id >= 1"]
    encoder = preprocessing.LabelEncoder()

    # 1、加载训练集
    train_text = SqlHelper().commonSelect(tableName=TRAIN_TABLE_NAME, params=["content"], conditions=conditions)
    train_text = [raw[0].decode() for raw in train_text]
    train_label = SqlHelper().commonSelect(tableName=TRAIN_TABLE_NAME, params=["type"], conditions=conditions)
    train_label = [raw[0] for raw in train_label]
    train_label = encoder.fit_transform(train_label)     # 按照训练集的label标签对所有样本的label进行编码

    # 2、使用卡方检验提取特征值。获取训练文本的特征矩阵，词袋
    # 提取的特征向量的大小
    max_features = 1000
    alpha = 1  # 平滑参数
    textVector, wordBag = createWordBag(train_text, train_label, max_features=max_features)

    #3、进行训练，计算先验概率
    train(textVector, train_label, alpha)
    # prioriProbability, textProportion = train(textVector, train_label,alpha)
    # print("用时：%f" % ((time.time() - ticks) / 60))

    # 4、预测
    # 加载测试集
    # encoder = preprocessing.LabelEncoder()
    # test_text = SqlHelper().commonSelect(tableName=TEST_TABLE_NAME, params=["content"], conditions=conditions)
    # test_text = [raw[0].decode() for raw in test_text]
    # test_label = SqlHelper().commonSelect(tableName=TEST_TABLE_NAME, params=["type"], conditions=conditions)
    # test_label = [raw[0] for raw in test_label]
    # test_label = encoder.fit_transform(test_label)  # 按照训练集的label标签对所有样本的label进行编码
    # predict(test_text,test_label,wordBag, prioriProbability, textProportion)
