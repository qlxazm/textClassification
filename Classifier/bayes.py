import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from SQLHelper.SqlHelper import SqlHelper
import time
from sklearn.utils import as_float_array
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import NuSVC
from sklearn.feature_extraction.text import TfidfTransformer



def load_stop_words(path):
    '''
    加载停用词表
    '''
    stop_words = []
    with open(path, 'r') as f:
        for word in f:
            stop_words.append(word)


def text_vector(text, bow=None):
    '''
    根据bow将文本转换成向量

    text:list[string]
    bow:按照词袋生成文本向量

    rtype：转换后的文本矩阵(ndarray)
    '''
    text_vecs = CountVectorizer(vocabulary=bow, dtype=np.uint8)
    # return np.array(text_vecs.fit_transform(text).todense(),dtype='np.uint8')
    return text_vecs.fit_transform(text)


def count_weight(X, bow=None):
    '''
    使用Tfidf算法进行权重计算
    X:raw_data
    bow：特征词袋

    rtype:list
    '''
    bow_weight = {}
    tfidf = TfidfVectorizer(max_df=maxDf, min_df=minDf, vocabulary=bow)
    tfIdf_train = tfidf.fit_transform(train_data)
    bow_weight = dict(bow, tfIdf_train)
    return tfidf, bow_weight


def chi2_select(X, y):
    '''
    使用卡方检验进行特征选择
    X:raw_data
    y:label
    dim:为选择的维度

    return:返回特征索引, 转换后的训练矩阵X
    '''
    text_vecs = CountVectorizer(dtype=np.uint8)
    X = text_vecs.fit_transform(X)

    # 使用TF-IDF进行加权
    transformer = TfidfTransformer()
    X = transformer.fit_transform(X)

    vocabulary = text_vecs.vocabulary_
    index2word = dict(zip(vocabulary.values(), vocabulary.keys()))  # 特征到词汇的索引

    scores, p_val = chi2(X, y)
    scores = as_float_array(scores, copy=True)
    scores[np.isnan(scores)] = np.finfo(scores.dtype).min
    # 从小到大选出来的索引
    indexs = np.argsort(scores, kind="mergesort")

    bow = []
    for index in indexs:
        bow.append(index2word[index])
    return indexs, bow, X


    # p_val_2_index = dict(zip(vocabulary.values(), p_val))
    # sorted_feat = sorted(p_val_2_index.items(), key = lambda x:x[1], reverse=True)
    # import pdb
    # pdb.set_trace()
    # tri_feat = []
    # for x in sorted_feat:
    #     tri_feat.append(tuple([x[0], x[1], index2word[x[0]]]))
    #
    # return tri_feat, bow
    #
    # feature_select = SelectKBest(chi2, k=dim)
    # feature_select.fit_transform(X, y)
    # # 生成所选特征的词袋
    # bow = []
    # indexs = []  # 选出的特征索引
    # for index in feature_select.get_support(indices=True):
    #     indexs.append(index)
    #     bow.append(index2word[index])
    # indexs = np.array(indexs)
    # type(X[:, indexs])
    # # X[:,indexs] 按索引列返回矩阵
    # return X[:, indexs], bow


class mulNaivebayes:
    def __init__(self):
        self.cat_num = None  # 分类数
        self.feature_num = None  # 特征数
        self.feature_fre = None  # 分类特征统计频数矩阵
        self.catFeature_sum = None  # 每一类的特征总数
        self.cat_sum = None  # 每一类的样本总数
        self.pre_cat = None  # 先验类别概率向量（取对数后）
        self.pre_prob = None  # 条件概率矩阵 （取对数后）
        self.post_prob = None  # 后验概率矩阵（取对数后）
        self.predictLable = None  # 预测标签

    def fit(self, X, y):
        '''
        采取多项式朴素贝叶斯算法
        X:train_data(ndarray)
        y:raw_label(list)
        '''
        # 标签转换为one-hot编码
        label_trans = LabelBinarizer()
        y = label_trans.fit_transform(y)
        self.cat_num = y.shape[1]

        sam_sum = y.shape[0]
        self.feature_num = X.shape[1]

        lam = 1  # 平滑因子,这里采用拉普拉斯平滑技术去掉概率为0的值

        # 特征在每一类别中出现的频数 row为标签 col为特征,value为类别i中出现特征j的频数
        self.feature_fre = y.T * X
        self.feature_fre = self.feature_fre + lam
        # 每一类的特征总数
        self.catFeature_sum = self.feature_fre.sum(axis=1)

        # 每一类别的初始概率p(y_j)
        self.cat_sum = y.sum(axis=0)
        self.pre_cat = np.log(self.cat_sum) - np.log(sam_sum)
        self.pre_cat.astype(dtype=np.float32)

        # 统计各类别中每一个单词出现的概率(p_wi_yj)
        self.pre_prob = np.zeros((self.cat_num, self.feature_num), dtype=np.float32)
        self.pre_prob = np.log(self.feature_fre) - np.log(self.catFeature_sum.reshape(-1, 1))

    def predict(self, X_test):
        '''
        return:
            np.array(the predicted label of test_sample)
        '''
        size = X_test.shape[0]
        # 后验概率，row为样本 col为标签 value为每一个样本在该分类标签下的后验概率
        self.predictLable = np.zeros(size)  # 预测标签结果
        # 判断两矩阵维度是否一致
        if X_test.shape[1] == self.feature_num:
            # self.post_prob = np.dot(X_test,
            #                         self.pre_prob.T) + self.pre_cat  # 由于转换为log，可以直接进行矩阵乘积,pre_prob需转置成行为词，列为label
            self.post_prob = X_test * self.pre_prob.T + self.pre_cat
        else:
            raise ValueError('dimension mismatch')
        for i in range(size):
            self.predictLable[i] = np.argmax(self.post_prob[i])

        return self.predictLable


def cross_validation(cls, x_train, y_train, k=10, average='micro'):
    '''
    自定义交叉验证函数

    cls:使用的算法模型
    x_train,y_train：转换好的训练数据
    k:表示将数据划分成几则
    average:表示采用何种评估方法，默认采用多分类的micro来计算f1值

    return: 返回平均准确度
    '''
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score

    skf = StratifiedKFold(n_splits=k)
    #     pre = []
    #     rec = []
    acc = []
    for train_index, test_index in skf.split(x_train, y_train):
        X_train, X_test = x_train[train_index], x_train[test_index]
        Y_train, Y_test = y_train[train_index], y_train[test_index]
        cls.fit(X_train, Y_train)
        label = cls.predict(X_test)
        #         pre.append(precision_score(Y_test,label,average=average))
        #         rec.append(recall_score(Y_test,label,average=average))
        acc.append(accuracy_score(Y_test, label))
    #     avg_precision = np.sum(pre)/k
    #     avg_recall = np.sum(rec)/k
    avg_accuacy = np.sum(acc) / k

    #     return (avg_precision,avg_recall,avg_accuacy)
    return avg_accuacy, cls


def grid_search(cls, bow_dim, x, y, k=10, average='micro'):
    '''
    网格搜索
    x:raw_data
    y:label_vec
    cls:分类器
    bow_dim:[param_1,...,param_n]，词袋维数选择
    k:交叉验证的分割数
    average:表示采用何种评估方法，默认采用多分类的micro来计算f1值
    '''
    score = []
    best_acc = 0.0
    best_param = 0
    best_bow = None
    best_clf = None

    feat_time_s = time.time()
    feat_sel_index, bow, X = chi2_select(x, y)
    feat_time_e = time.time()

    print("Feature selection time consuming :{}".format(feat_time_e - feat_time_s))
    for num in bow_dim:
        index = feat_sel_index[-num:]
        bow = bow[-num:]
        x_train = X[:, index]
        cross_time_s = time.time()
        acc,clf = cross_validation(cls, x_train, y, k=k, average=average)
        cross_time_e = time.time()
        print("accuracy:{}".format(acc))
        print("cross_validation time consuming :{}".format(cross_time_e-cross_time_s))

        score.append((num, acc))
        if acc > best_acc:
            best_param = num
            best_acc = acc
            best_bow = bow
            best_clf = clf

    return best_param, score, best_bow, best_clf


if __name__ == '__main__':
    train_table_name = "traindataset"
    test_table_name = "testdataset"
    conditions = ["id >= 1"]

    pl = preprocessing.LabelEncoder()
    # 加载训练集
    train_text = SqlHelper().commonSelect(tableName=train_table_name, params=["content"], conditions=conditions)
    train_text = [raw[0].decode() for raw in train_text]
    train_label = SqlHelper().commonSelect(tableName=train_table_name, params=["type"], conditions=conditions)
    train_label = [raw[0] for raw in train_label]
    train_label = pl.fit_transform(train_label)

    # 加载测试集
    test_text = SqlHelper().commonSelect(tableName=test_table_name,params=["content"],conditions=conditions)
    test_text = [raw[0].decode() for raw in test_text]
    test_label = SqlHelper().commonSelect(tableName=test_table_name,params=["type"],conditions=conditions)
    test_label = [raw[0] for raw in test_label]
    test_label = pl.transform(test_label)

    # cls = mulNaivebayes()
    #
    # bow_dim = [x for x in range(100000,200000,10000)]
    # best_param, score, best_bow, best_clf = grid_search(cls, bow_dim, train_text, train_label)


###################################################################################################################
    # SVM分类实现
    # 选择特征
    feat_sel_index, bow, X = chi2_select(train_text, train_label)
    # 特征数
    k = 1000
    X_train = X[:,feat_sel_index[-k:]]
    # 打乱数据
    X,y= shuffle(X,train_label,random_state=23333)

    #对数据进行归一化
    # scaler = StandardScaler(with_mean=False)
    # x_train = scaler.fit_transform(X)
    X_normalized = preprocessing.normalize(X, norm='l2')

    #分类器参数调优
    svm_clf = SGDClassifier(tol=0.01, n_jobs=2, shuffle=True, average=True)
    # svm_clf.fit(x_train,train_label)
    param_grid = [
        {'alpha':[0.0001, 0.001, 0.01, 0.1],
         'penalty':['l1','l2'],
         }
    ]
    gridSearch = GridSearchCV(svm_clf, param_grid, scoring='accuracy',cv=10)
    start = time.time()
    gridSearch.fit(X_normalized,y)

    #交叉验证后提取最佳参数
    bestAlpha = gridSearch.best_params_["alpha"]
    bestPenalty = gridSearch.best_params_["penalty"]
    bestEstimator = gridSearch.best_estimator_

    test_text = text_vector(test_text, bow)
    test_text,test_label = shuffle(test_text, test_label, random_state=23333)
    test_text = test_text[:100000]
    test_label = test_label[:100000]

    test_text_normalized = preprocessing.normalize(test_text, norm='l2')
    preLables = bestEstimator.predict(test_text_normalized)

    print(test_label[:100])
    print(preLables[:100])
    correct = 0
    index = 0
    for lable in preLables:
        if test_label[index] == lable:
            correct += 1
        index += 1
    print("correct rate is {0}".format(correct / len(test_label)))

    end = time.time()
    print("time consuming:{0} minutes".format((end - start) / 60))



