import cx_Oracle
import sqlalchemy
import pandas as pd

class DBConnection():
    def __init__(self, host, port, sid, creds):
        with open(creds) as credentials:
            username = credentials.readline().strip()
            password = credentials.readline().strip()
        self.connection = cx_Oracle.connect(username, password, cx_Oracle.makedsn(host, port, sid))
        # dialect+driver://username:password@host:port/database
        self.connectionStr = "oracle+cx_oracle://"+username+':'+password+'@'+host+':'+str(port)+'/'+sid

    def __del__(self):
        self.connection.close()

    def select(self, statement):
        cursor = self.connection.cursor()
        cursor.execute(statement)
        result = cursor.fetchall()
        labels = [row[0] for row in cursor.description]
        return pd.DataFrame(data=result, columns=labels)

    def insert(self, df, table, schema, columnTypes={}, ifExists="append"):
        engine = sqlalchemy.create_engine(self.connectionStr)
        df.to_sql(table, engine, schema=schema, if_exists=ifExists, dtype=columnTypes)
