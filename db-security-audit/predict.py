import pandas as pd
from joblib import load
from db import DBConnection

def loadData():
    connection = DBConnection("acc-db-05", 1541, "AUDT", ".credentials")
    query = open("update.sql", "r").read().replace(';', '')
    df = connection.select(query)
    df.set_index('SESSIONID', inplace=True)
    df.fillna(0, inplace=True)
    return df

def main():
    X = loadData()
    pipeline = load("model.joblib")
    distances = pipeline.transform(X)
    classifications = pipeline.predict(X)

    df = pd.DataFrame(index=X.index)
    for i in range(distances.shape[1]):
        df["CENTROID_DISTANCE_"+str(i)] = distances[:, i]
    df["CLASSIFICATION"] = classifications

    connection = DBConnection("acc-db-05", 1541, "AUDT", ".credentials")
    connection.insert(df, "sessions_anomaly_detection", "prod_user", ifExists="replace")

if __name__ == "__main__":
    main()