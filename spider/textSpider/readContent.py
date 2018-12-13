import pymysql
from textSpider.settings import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DBNAME,NEWS_TABLE
from SQLHelper.SqlHelper import SqlHelper

conn = pymysql.connect(MYSQL_HOST,MYSQL_USER,MYSQL_PASSWORD,MYSQL_DBNAME, charset='utf8')
cursor = conn.cursor()
url = 'http://finance.chinanews.com/it/2018/03-21/84.shtml'
sql =  " SELECT * FROM " + "it" + " where id=1"
cursor.execute(sql)
rows = cursor.fetchone()
if rows == None:
    print("没有记录")
for row in rows:
    print("id：" + row[0])
    print("内容：" + row[1].decode())
    print("类别：" + row[2])
cursor.close()
conn.close()

# sqlHelper = SqlHelper()
# sqlHelper.insert(tableName='test',record={"id":"3","content":"'插入第二条'"})

testList = [1,2,3,4]
j = 0
# while j < 4:
#     i = testList[j]
#     try:
#         if i == 3:
#             raise UnicodeDecodeError("utf8",bytes([1,2,3,4]),19,29,"OK")
#         print(i)
#     except UnicodeDecodeError:
#         print("捕获了异常")
#         j += 1
#         continue
#     print("正常的结束：%d".format(i))
#     j += 1

