import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from joblib import dump
from db import DBConnection

CAT_COLS = ["OS_USERNAME", "DB_USERNAME"]

def loadData():
    connection = DBConnection("acc-db-05", 1541, "AUDT", ".credentials")
    query = open("update.sql", "r").read().replace(';', '')
    df = connection.select(query)
    df.set_index('SESSIONID', inplace=True)
    df.fillna(0, inplace=True)
    return df

def createPipline(nClusters, catCols=CAT_COLS):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    pl = Pipeline([( "Transformer", ColumnTransformer( 
                                        [("Encoder", encoder, catCols)],
                                        remainder='passthrough' ) ),
                    ( "Scaler", MinMaxScaler() ),
                    ( "Model", KMeans(n_clusters=nClusters) )])
    return pl

def updateModel(df, fileName):
    pipeline = createPipline(3)
    pipeline.fit(df)
    dump(pipeline, fileName)

def main():
    df = loadData()
    updateModel(df, "model.joblib")

if __name__ == "__main__":
    main()