from SQLHelper.SqlHelper import SqlHelper
from sklearn import preprocessing

def data_load():

    train_table_name = "traindataset"
    test_table_name = "testdataset"
    conditions = ["id >= 1"]

    pl = preprocessing.LabelEncoder()
    # 加载训练集
    train_text = SqlHelper().commonSelect(tableName=train_table_name, params=["content"], conditions=conditions)
    train_text = [raw[0].decode() for raw in train_text]
    train_label = SqlHelper().commonSelect(tableName=train_table_name, params=["type"], conditions=conditions)
    train_label = [raw[0] for raw in train_label]
    train_label = pl.fit_transform(train_label)

    # 加载测试集
    test_text = SqlHelper().commonSelect(tableName=test_table_name, params=["content"], conditions=conditions)
    test_text = [raw[0].decode() for raw in test_text]
    test_label = SqlHelper().commonSelect(tableName=test_table_name, params=["type"], conditions=conditions)
    test_label = pl.fit_transform(test_label)

    return train_text,train_label,test_text,test_label