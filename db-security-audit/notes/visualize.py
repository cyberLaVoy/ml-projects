import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cluster, preprocessing
import pandas as pd
import numpy as np
import sys

def weekday2int(d):
    d = d.strip().lower()
    if d == "monday":
        return 1
    if d == "tuesday":
        return 2
    if d == "wednesday":
        return 3
    if d == "thursday":
        return 4
    if d == "friday":
        return 5
    if d == "saturday":
        return 6
    if d == "sunday":
        return 7

def stripDot(v):
    return int( str(v).replace('.', '') )

def dropExtra(df):
    #dropColumns = ["LOGON_HOST", "LOGON_PORT", "SYSTEM_PRIVILEGE_USED", "NON_SUCCESS_RETURN_CODE"]
    dropColumns = ["SYSTEM_PRIVILEGE_USED", "NON_SUCCESS_RETURN_CODE"]
    df.drop(columns=dropColumns, axis=1 , inplace=True)

def preprocess(df):
    df.fillna(0, inplace=True)
    df["LOGIN_DAY"] = df["LOGIN_DAY"].apply(weekday2int)
    df["LOGON_HOST"] = df["LOGON_HOST"].apply(stripDot) 
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(df.values)
    scaled = scaler.transform(df.values)
    df = pd.DataFrame(scaled, columns=df.columns, index=df.index)    
    return df

def visualize3D(values, labels, groups, colors):
    X, Y, Z = values
    plt.style.use("dark_background")
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X, Y, Z, c=groups, cmap=ListedColormap(colors))
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    plt.show() 

def findClusters(k, df):
    kMeans = cluster.KMeans(n_clusters=k)
    kMeans.fit(df)
    return kMeans.labels_, kMeans.cluster_centers_

def findOutliers(points):
    centroid = np.mean(points.T, axis=1)
    print(centroid)
    distances = np.linalg.norm( points-centroid , axis=1)
    outliers = distances > distances.mean()+4*distances.std()
    return outliers


def main():
    fin = sys.argv[1] 
    df = pd.read_csv(fin, index_col=0)
    df = preprocess(df)
    cols = list(df)
    
    dropExtra(df)
    cols = list(df)

    df["CLUSTER"] = 0
    outliers = findOutliers(df[cols].values)
    outliers = df[outliers].index
    print( list(outliers) )
    df.loc[outliers, "CLUSTER"] = 1

    #visualize3D(centers.T, cols, [i for i in range(k)], kColors)
    
    values = [df[cols[i]] for i in range(3)]
    visualize3D( values, cols, df["CLUSTER"], [(0,1,0,0), "#FF0000"])

main()

