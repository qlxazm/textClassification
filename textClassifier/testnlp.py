import nltk
import urllib.request
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
# 将文本tokennize成句子
from nltk.tokenize import sent_tokenize
# 将文本tokennize成单词
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from SQLHelper.SqlHelper import SqlHelper
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.metrics import classification_report

# response = urllib.request.urlopen("http://php.net")
# html = response.read()
# #取出html标签
# soup = BeautifulSoup(html, "html5lib")
# text = soup.get_text(strip=True)
# #将文本转化成token
# #返回的是list
# tokens = text.split()
#
# #去除停用词
# clean_tokens = list()
# sr = stopwords.words("english")  #获取停用词表
# for token in tokens:
#     if token not in sr:
#         clean_tokens.append(token)
#
# #统计词频
# freq = nltk.FreqDist(clean_tokens)
# for key,val in freq.items():
#     print(str(key) + ":" + str(val))
# #根据词频画图
# # freq.plot(20, cumulative=False)

# train_table_name = "traindataset"
# test_table_name = "testdataset"
# conditions = ["id >= 1"]
#
# encoder = preprocessing.LabelEncoder()
# # 1、加载训练集
# train_text = SqlHelper().commonSelect(tableName=train_table_name, params=["content"],conditions = conditions)
# train_text = [raw[0].decode() for raw in train_text]
# train_label = SqlHelper().commonSelect(tableName=train_table_name, params=["type"], conditions = conditions)
# train_label = [raw[0] for raw in train_label]
# # 按照训练集的label标签对所有样本的label进行编码
# train_label = encoder.fit_transform(train_label)
# # 2、使用卡方检验提取特征值。获取训练文本的特征矩阵，词袋
# #提取的特征向量的大小
# max_features=1000
# #使用m估计时使用到的p值
# p = 1
# textVector, wordBag = mybayes.createWordBag(train_text, train_label, max_features=max_features)
# mybayes.train(textVector, train_label)


# vectorizer = CountVectorizer()
# corpus = [
#     "I come to China to travel",
#     "This is a car polupar in China",
#     "I love tea and Apple ",
#     "The work is to write some papers in science"
# ]
# """
# 下面这一行的输出类似：(3, 8)	1
# 其中左边括号中的第一个数字是文本的序号
# 第二个数字是词的序号，注意词的序号是基于所有的文档的
# 第三个数字是统计出来的词频
# """
# print(vectorizer.fit_transform(corpus))
#基于词袋模型对文本进行向量化之后的词向量，因为文本的的向量大部分是0，所以实践中更多的是用稀疏矩阵进行存储
#将文本做了词频统计之后，可以使用TF-IDF对词特征值修正
#如果向量的维度过高，可以使用Hash Trick进行降维
# print(vectorizer.fit_transform(corpus).toarray())
#可以查看特征向量中每个特征代表的词
# print(vectorizer.get_feature_names())

#使用TF-IDF值   方法1：先向量化之后再计算TF-IDF值
# transformer = TfidfTransformer()
# tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
# print(tfidf)
#
# #使用TF-IDF值   方法2：使用TfidfVectorizer向量化并计算TF-IDF值
# tfidf2 = TfidfVectorizer()
# re = tfidf2.fit_transform(corpus)
# print("\n")
# print(re)



# import jieba
# #向jieba中加入需要特殊处理的词
# jieba.suggest_freq('沙瑞金', True)
# jieba.suggest_freq('易学习', True)
# jieba.suggest_freq('王大路', True)
# jieba.suggest_freq('京州', True)
#
# with open('./text2.txt','r', encoding="UTF-8") as f:
#     document_decode = f.read()
#     document_cut = jieba.cut(document_decode)
#     result = " ".join(document_cut)
#     with open('./cuted2.txt', 'w') as f1:
#         f1.write(result)
#     f1.close()
# f.close()
#从文件导入停用词表
# stopwordspath = "./stop_words.txt"
# stopwords_dict = open(stopwordspath, 'rb')
# stopwords_content = stopwords_dict.read()
# stopwordslist = stopwords_content.splitlines()
# stopwords_dict.close()
# #将已经分好词的文本载入内存并进行向量化
# with open('./cuted1.txt') as f1:
#     res1 = f1.read()
# with open("./cuted2.txt") as f2:
#     res2 = f2.read()
# corpus = [res1,res2]
# vector = TfidfVectorizer(stop_words=stopwordslist) #使用停用词表
# tfidf = vector.fit_transform(corpus)
# print(tfidf)

#
# clf = joblib.load("train_model.m")
#
# try:
#     clf = joblib.load("train_model.m")
# except FileNotFoundError:


le = preprocessing.LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])

print(le.classes_)

# wordbag = ["你好","我好"]
# with open("wordBag.txt","w") as f:
#     f.write(" ".join(wordbag))
# f.close()

with open("wordBag.txt","r") as f:
    line = f.readline()
    print(line.split(" "))
f.close()
