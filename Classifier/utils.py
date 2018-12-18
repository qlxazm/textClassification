import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import as_float_array


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


