# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class TextspiderItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass

#定义转存到mysql的item
class NewsItem(scrapy.Item):
    content = scrapy.Field()
    type = scrapy.Field()
    url = scrapy.Field()

