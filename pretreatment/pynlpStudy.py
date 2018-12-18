# -*- coding: utf-8 -*-
import pymysql
from SQLHelper.SqlHelper import SqlHelper
from pretreatment.utils import wordSegmenter
#批量插入的条数
BATCH_SIZE=3000

sourceTable=["tynews","fortunenews", "autonews", "estatenews", "milnews", "jknews", "lifenews", "itnews", "whnews", "ylnews"]
destTable=["ty","fortune","auto","estate","mil", "jk", "life", "it", "wh", "yl"]

i = 0
while i < len(sourceTable):
    TABLE_NAME = sourceTable[i]
    DEST_TABLE_NAME = destTable[i]

    sqlHelper = SqlHelper()
    sqlHelper1 = SqlHelper()
    sqlHelper.commonSelect(tableName=TABLE_NAME, params=["content", "type"])

    row = sqlHelper.getOneRecord()
    while row != None:
        args = []
        num = 0
        #查询出批量数据
        while num < BATCH_SIZE and row != None:
            content = row[0].decode()
            typeName = row[1]
            # 捕获UnicodeDecodeError错误并跳过
            try:
                """
                分词并去除停用词
                """
                content = wordSegmenter(content)
            except UnicodeDecodeError:
                row = sqlHelper.getOneRecord()
                continue

            if content != '':
                args.append([pymysql.escape_string(content),typeName])
                num += 1
            row = sqlHelper.getOneRecord()
        #批量插入
        sql = sqlHelper1.insertMany(tableName=DEST_TABLE_NAME,params=["content", "type"],args=args)
        if row != None:
            row = sqlHelper.getOneRecord()
    sqlHelper.close()
    sqlHelper1.close()
    i+=1








