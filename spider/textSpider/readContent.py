import pymysql
from textSpider.settings import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DBNAME,NEWS_TABLE

conn = pymysql.connect(MYSQL_HOST,MYSQL_USER,MYSQL_PASSWORD,MYSQL_DBNAME, charset='utf8')
cursor = conn.cursor()
url = 'http://finance.chinanews.com/it/2018/03-21/8473025.shtml'
sql =  " SELECT * FROM " + "tynews" + " where url='" + url + "'"
cursor.execute(sql)
rows = cursor.fetchall()
print(len(rows))
for row in rows:
    print("内容：" + row[0].decode())
    print("类别：" + row[1])
    print("url：" + row[2])
cursor.close()
conn.close()