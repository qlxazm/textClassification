# -*- coding: utf-8 -*-
"""
爬取中国新闻网站新闻，其中IT类入口是：http://www.chinanews.com/scroll-news/it/2018/1123/news.shtml
"""
import scrapy
import time
import logging
from textSpider.settings import SPIDER_CATEGORY
from textSpider.items import NewsItem

class NewsSpider(scrapy.Spider):
    name = "NewsSpider"
    # allowed_domains = "chinanews.com"

    def __init__(self, startDate = None,endDate="2018/11/20",*args,**kwargs):
        super(NewsSpider,self).__init__(*args,**kwargs)
        # 解析要爬取的数据的终止时间
        if len(endDate.split("/")) != 3:
            endDate = "2018/11/20"
        endDateList = endDate.split("/")
        endYear = int(endDateList[0])
        endMonth = int(endDateList[1])
        endDay = int(endDateList[2])

        # 如果设置了开始时间
        if startDate is not None:
            localtime = startDate.split("/")
        else: #没有设置开始时间，就以当前时间的前一天为开始时间
            localtime = time.localtime(time.time() - 86400)
        startYear = int(localtime[0])
        startMonth = int(localtime[1])
        startDay = int(localtime[2])

        #计算开始日期到结束日期的日期字符串
        dateStr = self.parseDate(startYear=startYear,startMonth=startMonth,startDay=startDay,
                       endYear=endYear,endMonth=endMonth,endDay=endDay)
        # 目前要爬取的栏目种类是SPIDER_CATEGORY类
        urlPrefix = "http://www.chinanews.com/scroll-news/" + SPIDER_CATEGORY + "/"
        urlSuffix = "/news.shtml"
        print("开始时间：" + str(startYear) + "/" + str(startMonth) + "/" + str(startDay))
        print("结束时间: " + endDate)
        self.start_urls   = [urlPrefix + dateItem + urlSuffix for dateItem in dateStr]
    # @classmethod
    # def from_crawler(cls, crawler, *args, **kwargs):
    #     spider = super(NewsSpider, cls).from_crawler(crawler, *args, **kwargs)
    #     # 定义爬虫开启的钩子
    #     crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
    #     # 定义爬虫关闭的钩子
    #     crawler.signals.connect(spider.spider_opened, signal=signals.spider_opened)
    #     return spider
    #
    # def spider_closed(self, spider):
    #     spider.logger.info('Spider closed: %s', spider.name)
    #
    # def spider_opened(self, spider):
    #     spider.logger.info('Spider closed: %s', spider.name)

    def parse(self, response):
        urls = response.xpath("//div[@class='content_list']//a/@href").extract()
        if urls is not None:
            for url in urls:
                yield scrapy.Request(url=url, callback=self.parseNew)

    def parseNew(self, response):
        content = ""
        paragraphs = response.xpath("//div[@class='left_zw']/p")
        separate = ''
        for paragraph in paragraphs:
            paragraph = paragraph.xpath('text()').extract()
            content += separate.join(paragraph)
        # content就是最后爬取的内容，返回NewsItem
        yield NewsItem(content=content, type = SPIDER_CATEGORY, url = response.url)


    def parseDate(self,*,startYear,startMonth,startDay,endYear,endMonth,endDay):
        """
        计算从结束年月日到开始年月日的日期字符串，eg:2018/1111。注意：月份和天数是连在一起的
        :param startYear:
        :param startMonth:
        :param startDay:
        :param endYear:
        :param endMonth:
        :param endDay:
        :return: list
        """
        #月份天数对照表，第一行平年，第二行闰年
        daysOfMonth = [[0,31,28,31,30,31,30,31,31,30,31,30,31],
                       [0,31,29,31,30,31,30,31,31,30,31,30,31]]
        result = []
        while endYear != startYear or endMonth != startMonth or endDay != startDay:
            dateStr = str(startYear) + "/"
            if startMonth < 10:
                dateStr += "0" + str(startMonth)
            else:
                dateStr += str(startMonth)
            if startDay < 10:
                dateStr += "0" + str(startDay)
            else:
                dateStr += str(startDay)
            result += [dateStr]
            startDay -= 1
            if startDay == 0:
                startMonth-=1
                if startMonth == 0:
                    startMonth = 12
                    startYear -= 1
                leapYear = self.isLeapYear(startYear)
                startDay = daysOfMonth[leapYear][startMonth]
        return result

    def isLeapYear(self,year):
        """
        判断平年闰年，闰年返回1，平年返回0
        :param year:
        :return:
        """
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return 1
        else:
            return 0


