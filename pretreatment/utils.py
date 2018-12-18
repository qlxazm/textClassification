# -*- coding: utf-8 -*-
"""
存放一些工具函数
"""
import pynlpir
from os import path
import re
#这个文件不算文件名在内的路径
ROOT = path.dirname(__file__)
#默认的停用词
STOP_WORDS = "stopWords.txt"

def wordSegmenter(sentence='',pathOfStopWords=''):
    """
    将传入的句子分词并去除停用词
    :param sentence:         传入的句子
    :param pathOfStopWords:  停用词的路径
    :return:                 分词并去除停用词后由空格分隔的字符串
    """
    #打开分词器
    pynlpir.open()
    #分词
    seg_list = []
    for seg in pynlpir.segment(sentence):
        seg_list.append(seg[0])
    #去除停用词
    resultWords = []
    if pathOfStopWords == '': #没指定停用词就使用默认的停用词
        pathOfStopWords = path.join(ROOT,STOP_WORDS)
    f_stop = open(pathOfStopWords, 'rt', encoding='utf-8')
    try:
        f_stop_text = f_stop.read()
    finally:
        f_stop.close()
    f_stop_words = f_stop_text.split("\n")
    for seg in seg_list:
        seg = seg.strip()
        if re.match(r'[a-zA-Z0-9]+',seg): #去掉英文以及数字
            continue
        if len(seg) > 0 and (seg not in f_stop_words):
            resultWords.append(seg)
    return " ".join(resultWords)

