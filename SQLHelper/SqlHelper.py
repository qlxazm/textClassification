# -*- coding: utf-8 -*-
import pymysql

class SqlHelper:
    #连接mysql数据库的属性，这里定义成类属性
    MYSQL_HOST = "localhost"
    MYSQL_USER = "root"
    MYSQL_PASSWORD = "qlxazm"
    MYSQL_DBNAME = "mydb"

    def __init__(self):
        """
        创建链接属性和游标
        """
        self._conn = pymysql.connect(SqlHelper.MYSQL_HOST,
                                     SqlHelper.MYSQL_USER,
                                     SqlHelper.MYSQL_PASSWORD,
                                     SqlHelper.MYSQL_DBNAME,
                                     charset='utf8')
        self._cursor = self._conn.cursor()

    def commonSelect(self, tableName = '', params = [], conditions = []):
        """
        执行常用的SQL查询，然后可以调用getOneRecord方法返回一条记录
        :param tableName:  数据库表名
        :param params:     要查询的记录属性，比如['conteny','id']
        :param conditions: 要查询记录的条件, 比如['id=1','type=it']
        :return:           无
        """
        #解析参数字符串
        paramStr = ",".join(params)
        #没有提供属性表示读取所有属性
        if len(paramStr) == 0:
            paramStr = "*"

        sqlStr = "SELECT {0} FROM {1} ".format(paramStr, tableName)

        #解析条件字符串
        conditionsStr = " AND ".join(conditions)
        if len(conditionsStr) > 0:
            sqlStr += " WHERE {0}".format(conditionsStr)
        self._cursor.execute(sqlStr)
        return self._cursor.fetchall()

    def insert(self, tableName='',record={}):
        """
        向表中插入一条记录
        :param tableName:
        :param record: 以字典形式表示的记录，比如：{"id":1,"content":"'这是字符串值，必须多加一个单引号才能插入数据库'"}，在传入字符串时一定要多加一个单引号
        :return:       插入成功返回True,否则 False
        """
        params = list(record.keys())
        #如果没有传入要插入的记录，则返回False
        if len(params) == 0 or tableName == '':
            return False
        values = []
        for i in params:
            values.append(record[i])
        sqlStr = "INSERT INTO {0}({1}) VALUES({2})".format(tableName, ",".join(params), ",".join(values))
        try:
            #提交到数据库
            result = self._cursor.execute(sqlStr)
            self._conn.commit()
            if result == 1:
                return True
            else:
                return False
        except Exception as e:
            self._conn.rollback()
            print(e)
            return False

    def insertMany(self, tableName='',params=[],args=[]):
        """
        批量插入
        :param tableName:
        :param params:      要批量插入到的属性列
        :param args:        批量插入的记录组成的list，记录的类型是list
        :return:            批量插入的条数
        """
        if len(args) == 0:
            return
        if len(params) == 0:
            return
        if len(args[0]) != len(params):
            return

        sql = "INSERT INTO {0}({1}) ".format(tableName, ",".join(params))
        #占位符字符串
        placeHolder = ("%s." * len(params)).split(".")
        placeHolder = ",".join(placeHolder)
        placeHolder = placeHolder[:-1]

        sql += " VALUES({0})".format(placeHolder)
        result = self._cursor.executemany(sql, args)
        self._conn.commit()
        return result


    def getOneRecord(self):
        """
        返回一条记录
        :return:
        """
        if self._cursor == None:
            return None
        return self._cursor.fetchone()

    def close(self):
        if self._cursor != None:
            self._cursor.close()
        if self._conn != None:
            self._conn.close()


