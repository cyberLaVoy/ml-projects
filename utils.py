import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy import stats
from custom_estimators import MissingValuesPopulator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from joblib import dump, load
from tensorflow.keras.models import load_model


def saveModel(model, fileName, modelType):
    if modelType =="reg":
        regularSave(model, fileName)
    elif modelType == "net":
        saveNeuralNetworkModel(model, fileName)
    elif modelType == "vote":
        saveVotingNets(model, fileName)

def loadModel(fileName, modelType):
    if modelType =="reg":
        return regularLoad(fileName)
    elif modelType == "net":
        return loadNeuralNetworkModel(fileName)
    elif modelType == "vote":
        return loadVotingNets(fileName)


def regularSave(model, fileName):
    dump(model, fileName + ".joblib")

def regularLoad(fileName):
    return load(fileName + ".joblib")

def saveNeuralNetworkModel(pipeline, fileName):
    pipeline.named_steps["Model"].model.save(fileName + "_network.h5", include_optimizer=False)
    pipeline.named_steps["Model"].model = None
    dump(pipeline, fileName + "_network.joblib")

def loadNeuralNetworkModel(fileName):
    pipeline = load(fileName + "_network.joblib")
    pipeline.named_steps["Model"].model = load_model(fileName + "_network.h5")
    return pipeline


def saveVotingNets(pipeline, fileName):
    networks = pipeline.named_steps["Model"].named_estimators_
    for name in networks:
        if "net" in name:
            networks[name].model.save(fileName + name + ".h5", include_optimizer=False)
            networks[name].model = None
    dump(pipeline, fileName + "_votingNetwork.joblib")

def loadVotingNets(fileName):
    pipeline = load(fileName + "_votingNetwork.joblib")
    networks = pipeline.named_steps["Model"].named_estimators_
    for name in networks:
        if "net" in name:
            networks[name].model = load_model(fileName + name + ".h5")
    return pipeline






def getCategoricalColumnsInfo(X):
    missing = MissingValuesPopulator()
    missing.fit(X)
    X = missing.transform(X)

    categoricalColumns = list(X.select_dtypes(include=[np.object]))
    transformer = ColumnTransformer( [("encoder", OneHotEncoder(), categoricalColumns)] )
    transformer.fit(X)
    categories = transformer.named_transformers_["encoder"].categories_ 
    return categoricalColumns, categories

def examineNewVariables(X, y, var1, var2):
    print("var1 correlation:", stats.pearsonr(X[var1], y))
    print("var2 correlation:", stats.pearsonr(X[var2], y))
    newVar = var1 + 'AND' + var2 + " - Above Avg"
    newColumn = X.apply(lambda row: row[var1] >= X[var1].mean() and row[var2] >= X[var2].mean(), axis=1)
    print(newVar + "correlation:", stats.pearsonr(newColumn, y))

def visualize3D(predictors, labels, x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(predictors[x], predictors[y], predictors[z], c=labels, cmap=ListedColormap(['#0000FF', '#FF0000']), edgecolors='k')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.show() 

def viewConfusionMatrix(yActual, yPredicted):
    confMatrix = confusion_matrix(yActual, yPredicted)
    TN = confMatrix[0][0]
    FP = confMatrix[0][1]
    FN = confMatrix[1][0]
    TP = confMatrix[1][1]
    print("Confusion matrix:")
    print(confMatrix)
    print("Recall/Sensitivity Score:", TP/(TP+FN))
    print("Precision Score:", TP/(TP+FP))
    print("Specificty Score", TN/(TN+FP))
    print("Accuracy Score", (TP+TN)/(TP+TN+FP+FN))


def calcStatsInfo(X, y):
    missing = MissingValuesPopulator()
    missing.fit(X)
    X = missing.transform(X)
    categoricalColumns, categories = getCategoricalColumnsInfo(X)
    numericalHeaders = list(X.select_dtypes(exclude=[np.object]))
    transformer = ColumnTransformer( [("encoder", OneHotEncoder(categories=categories,sparse=False), categoricalColumns)], remainder="passthrough" ) 
    X = transformer.fit_transform(X)

    correlations = [stats.pearsonr(col, y) for col in X.T]
    pearsonr = np.array([v[0] for v in correlations])
    personsp = np.array([v[1] for v in correlations])
    anovaf, anovap = f_classif(X, y)
    informationGain = mutual_info_classif(X, y) 

    headers = []
    for i in range(len(categories)):
        col = categoricalColumns[i]
        for cat in categories[i]:
            headers.append(col + ' - ' + cat)
    headers.extend(numericalHeaders)

    statsInfo = pd.DataFrame({"FEATURE" : headers, "PEARSONS_R": pearsonr, "PEARSONS_P": personsp, "ABS_PEARSONS_R": np.absolute(pearsonr), "PEARSONS_R2": np.power(pearsonr, 2), "ANOVA_F" : anovaf , "ANOVA_P": anovap, "INFORMATION_GAIN": informationGain})
    statsInfo.set_index(keys=["FEATURE"], inplace=True)
    statsInfo.sort_values("ABS_PEARSONS_R", ascending=False, inplace=True)
    return statsInfo

def crossEvaluateModel(pipeline, predictors, labels, metrics):
    missing = MissingValuesPopulator()
    predictors = missing.transform(predictors)
    results = {}
    for metric in metrics:
        scores = cross_val_score(pipeline, predictors, labels, cv=3, scoring=metric)
        results[metric] = (scores.mean(), scores.std())
    return results


def levelClassBias(df, label):
    count1, count2 = df[label].value_counts()
    if count1 > count2:
        sample = df[df[label] == 0].sample(count1-count2, random_state=42)
    else:
        sample = df[df[label] == 1].sample(count2-count1, random_state=42)
    df = df.append( sample )
    return df.sample(frac=1, random_state=42)