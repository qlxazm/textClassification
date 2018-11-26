# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

from textSpider.settings import MYSQL_HOST, MYSQL_PASSWORD, MYSQL_DBNAME, MYSQL_USER, NEWS_TABLE
import pymysql

class TextspiderPipeline(object):
    def process_item(self, item, spider):
        """
        处理每个item，将其存入数据库
        :param item:
        :param spider:
        :return:
        """
        if self.conn is None:
            return
        if self.cursor is None:
            return
        sql = "INSERT INTO " + NEWS_TABLE + " (content, type, url) VALUES('{0}','{1}', '{2}')".format(item["content"],item["type"],item["url"])
        self.cursor.execute(sql)
        self.conn.commit()
        return item
    def open_spider(self, spider):
        """
        创建数据库连接并保存为对象的属性，open_spider只在spider运行期间运行一次
        :param spider:
        :return: 无
        """
        self.conn = pymysql.connect(MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DBNAME, charset='utf8')
        self.cursor = self.conn.cursor()
    def close_spider(self, spider):
        """
        关闭数据库连接，close_spider只在spider关闭时运行一次
        :param spider:
        :return: 无
        """
        if self.cursor is not None:
            self.cursor.close()
            self.conn.close()

