from db import DBConnection
from sqlalchemy import String, Integer, Float


def main():

    connection = DBConnection("acc-db-05", 1541, "AUDT", ".credentials")

    query = open("extract.sql", "r").read().replace(';', '')
    df = connection.select(query)
    df.set_index('SESSIONID', inplace=True)

    columnTypes = {"SESSIONID": Integer(),
                   "TIME_LOGGED_IN": Float(),
                   "ACTION_COUNT": Integer(),
                   "LOGIN_DAY": String(),
                   "LOGIN_HOST": String(),
                   "LOGIN_PORT": Integer(),
                   "DB_USERNAME": String(),
                   "OS_USERNAME": String(),
                   "SYSTEM_PRIVILEGE_USED": Integer(),
                   "NON_SUCCESS_RETURN_CODE": Integer()}
    connection.insert(df, "sessions_summaries", "prod_user", columnTypes=columnTypes)

if __name__ == "__main__":
    main()