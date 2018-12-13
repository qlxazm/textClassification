# -*- coding: utf-8 -*-
import pymysql
from SQLHelper.SqlHelper import SqlHelper
from textVectorGenerator.utils import wordSegmenter
#链接数据库的属性
TABLE_NAME = 'itnews'
DEST_TABLE_NAME = 'it'
#批量插入的条数
BATCH_SIZE=3000
#  ,"tynews","fortunenews", "autonews", "estatenews"
#  ,"ty","fortune","auto","estate"
sourceTable=["mil_temp_news"]
destTable=["mil"]

i = 0
while i < 1:
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








