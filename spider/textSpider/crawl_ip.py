import requests #用requests库来做简单的网络请求
import pymysql
from scrapy.selector import Selector
from textSpider.settings import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DBNAME,MYSQL_TABLE

conn = pymysql.connect(MYSQL_HOST,MYSQL_USER,MYSQL_PASSWORD,MYSQL_DBNAME, charset='utf8')
cursor = conn.cursor()

def clear_table():
    """
    清空存放代理IP的数据库的内容
    :return: 无
    """
    cursor.execute("truncate table " + MYSQL_TABLE)
    conn.commit()

def crawl_proxy_ip(pages):
    """
    爬取pages页代理ip的信息，存入mysql数据库
    :param pages: 一共要爬取多少页
    :return: 无
    """
    clear_table()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:52.0) Gecko/20100101 Firefox/52.0"}
    for i in range(1,pages):
        response = requests.get(
            url= 'http://www.xicidaili.com/nn/{0}'.format(i),
            headers=headers
        )
        all_trs = Selector(text=response.text).css("#ip_list tr")
        ip_list = []
        #提取一页的代理ip
        for tr in all_trs[1:]:
            ip = tr.xpath('td[2]/text()').extract_first()
            port = tr.xpath('td[3]/text()').extract_first()
            ip_type = tr.xpath('td[6]/text()').extract_first()
            ip_speed = tr.xpath('td[7]/div/@title').extract_first()
            if ip_speed:
                ip_speed = float(ip_speed.split(u'秒')[0])
            ip_alive = tr.xpath('td[9]/text()').extract_first()
            ip_list.append((ip, port, ip_type, ip_speed, ip_alive))
            # print("ip: %s, port: %s, type: %s, speed: %f, alive: %s",(ip, port, ip_type, ip_speed, ip_alive))
        #提取完一页的代理ip之后，存入mysql
        for ip_info in ip_list:
            sql = "insert into " + MYSQL_TABLE + "(ip, port, type, speed, alive)" \
                  + " VALUES('{0}','{1}','{2}','{3}','{4}')".format(ip_info[0],ip_info[1],ip_info[2],ip_info[3],ip_info[4])
            #print(sql)
            cursor.execute(sql)
            conn.commit()

class IPUtil(object):
    def get_random_ip(self):
        """
        随机获取一个可用的代理ip
        :return: 可用的代理ip
        """
        #从数据库随机获取一个ip
        sql = "SELECT ip, port, type FROM " + MYSQL_TABLE + " ORDER BY RAND() LIMIT 1"
        result = cursor.execute(sql)
        for ip_info in cursor.fetchall():
            ip = ip_info[0]
            port = ip_info[1]
            ip_type = ip_info[2]
            judge_re = self.judge_ip(ip, port, ip_type)
            if judge_re:
                ips = "{2}://{0}:{1}".format(ip, port, str(ip_type).lower())
                print("可用的ip代理" + ips)
                return  ips
            else:
                return self.get_random_ip()

    def judge_ip(self, ip, port, ip_type):
        """
        通过向百度发送请求来判断代理ip是否可用
        :param ip:
        :param port:
        :param ip_type:
        :return: 代理ip可用，返回true；否则，false
        """
        http_url = "https://www.baidu.com"
        proxy_url = "{2}://{0}:{1}".format(ip, port, str(ip_type).lower())
        try:
            proxy_dict = {
                "http": proxy_url
            }
            response = requests.get(http_url, proxies=proxy_dict)
        except Exception as e:
            print("无效IP地址，不能连接到百度")
            self.delete_ip(ip)
            return False
        else:
            code = response.status_code
            if code >= 200 and code < 300:
                print("有效IP")
                return True
            else:
                print("无效IP地址，不能连接到百度")
                self.delete_ip(ip)
                return False

    def delete_ip(self, ip):
        """
        删除ip
        :param ip:
        :return: 输出成功返回True
        """
        sql = "DELETE FROM " + MYSQL_TABLE + " WHERE ip = '{0}'".format(ip)
        cursor.execute(sql)
        conn.commit()
        return True

# if __name__ == '__main__':
    # 获取3页代理ip
    # crawl_proxy_ip(4)
    #关闭数据库连接
    # conn.close()
    # print(IPUtil().delete_ip('101.236.57.214'))
