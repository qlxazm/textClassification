# -*- coding: utf-8 -*-
from SQLHelper.SqlHelper import SqlHelper

CONDITIONS = ["id >= 1"]

def loadDataSet(tableName=None, conditions=CONDITIONS):
    """
    加载数据
    :param trainTableName:
    :param conditions:
    :return:
    """
    text = SqlHelper().commonSelect(tableName=tableName, params=["content"], conditions=conditions)
    text = [raw[0].decode() for raw in text]
    label = SqlHelper().commonSelect(tableName=tableName, params=["type"], conditions=conditions)
    label = [raw[0] for raw in label]
    return text, label