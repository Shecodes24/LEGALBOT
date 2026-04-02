import pymysql

def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="",
        port=3308,
        database="law_records",
        cursorclass=pymysql.cursors.DictCursor
    )
