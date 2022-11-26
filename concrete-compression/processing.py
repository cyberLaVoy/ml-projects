#!/usr/bin/env python3
import math
from prettytable import PrettyTable
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, cross_val_score
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel

# Info with headers: Cement (component 1)(kg in a m^3 mixture),Blast Furnace Slag (component 2)(kg in a m^3 mixture),Fly Ash (component 3)(kg in a m^3 mixture),Water  (component 4)(kg in a m^3 mixture),Superplasticizer (component 5)(kg in a m^3 mixture),Coarse Aggregate  (component 6)(kg in a m^3 mixture),Fine Aggregate (component 7)(kg in a m^3 mixture),Age (day), Concrete compressive strength(MPa, megapascals)
# Just headers = Cement,Blast Furnace Slag,Fly Ash,Water,Superplasticizer,Coarse Aggregate,Fine Aggregate,Age (days),Concrete Compressive Strength
DATA_FILENAME = "compresive_strength_concrete.csv"
PREDICTORS_HEADERS = ["Cement","Blast Furnace Slag","Fly Ash","Water","Superplasticizer","Coarse Aggregate","Fine Aggregate","Age (days)"]
LABEL_NAME = "Concrete Compressive Strength"

def removeOutliers(dataFrame):
        #print(X[ ( X[PREDICTORS_HEADERS[6]] >= 930 ) ]) # use to view amounts given cut off
        withCuts = dataFrame[ ( dataFrame[PREDICTORS_HEADERS[0]] >= 130 ) & ( dataFrame[PREDICTORS_HEADERS[0]] <= 550 ) 
                            & ( dataFrame[PREDICTORS_HEADERS[1]] >= 0 ) & ( dataFrame[PREDICTORS_HEADERS[1]] <= 325 ) 
                            & ( dataFrame[PREDICTORS_HEADERS[2]] >= 0 ) & ( dataFrame[PREDICTORS_HEADERS[2]] <= 215 )
                            & ( dataFrame[PREDICTORS_HEADERS[3]] >= 135 ) & ( dataFrame[PREDICTORS_HEADERS[3]] <= 230 )
                            & ( dataFrame[PREDICTORS_HEADERS[4]] >= 0 ) & ( dataFrame[PREDICTORS_HEADERS[4]] <= 17.5 )
                            & ( dataFrame[PREDICTORS_HEADERS[5]] >= 800 ) & ( dataFrame[PREDICTORS_HEADERS[5]] <= 1150 )
                            & ( dataFrame[PREDICTORS_HEADERS[6]] >= 550 ) & ( dataFrame[PREDICTORS_HEADERS[6]] <= 930 )
                            & ( dataFrame[PREDICTORS_HEADERS[7]] >= 0 ) & ( dataFrame[PREDICTORS_HEADERS[7]] <= 375 )
                            ]
        return withCuts

def outlierExporation(dataFrame):
    transDataFrame = removeOutliers(dataFrame)
    displayHistograms(transDataFrame) # display all histograms
    #displayScatterPlots(transDataFrame[PREDICTORS_HEADERS], transDataFrame[LABEL_NAME])
    #displayHistograms(pd.DataFrame(transDataFrame[PREDICTORS_HEADERS[3]]), 50) # display one feature at a time

def loadData(fileName=DATA_FILENAME):
    return pd.read_csv(fileName)
def saveData(df, fileName):
    df.to_csv(fileName)
def trainTestSplit(dataFrame, testRatio, sameSplit=True): # non-stratified split
    if sameSplit:
        dataTrain, dataTest = train_test_split(dataFrame, test_size=testRatio, random_state=42) # set random_state to always get the same "random" data
    else:
        dataTrain, dataTest = train_test_split(dataFrame, test_size=testRatio)
    return dataTrain, dataTest
def trainTestSplitStratified(dataFrame, stratFeatureName, newCatName, testRatio, catMin, catMax, catDivisor):
    dataFrame[newCatName] = np.ceil(dataFrame[stratFeatureName] / catDivisor)
    dataFrame[newCatName].where(dataFrame[newCatName] > catMin, catMin, inplace=True)
    dataFrame[newCatName].where(dataFrame[newCatName] < catMax, catMax, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=testRatio, random_state=42)
    for trainIndex, testIndex in split.split(dataFrame, dataFrame[newCatName]):
        stratTrainData = dataFrame.loc[trainIndex]
        stratTestData = dataFrame.loc[testIndex]
    return stratTrainData.drop(newCatName, axis=1), stratTestData

def predictorsLabelsSeparation(dataFrame, labelName=LABEL_NAME):
    predictors = dataFrame.drop(labelName, axis=1)
    labels = dataFrame[labelName].copy()
    return predictors, labels

# basic histogram display with Pandas dataframe
def displayHistograms(dataFrame, numBins=50):
    dataFrame.hist(bins=numBins, log=True, figsize=(20,15))
    #plt.savefig( "histograms.pdf" ) # if you wish to save the plot to a .pdf
    plt.show() # must have user-specified graphical backend

def displayScatterPlots( predictors, labels, logScalePredictors=True): # preditors as pandas dataframe; labels are numpy array
    predictorsHeaders = list(predictors) 
    numColumns = predictors.shape[ 1 ]
    shapeRows = math.ceil( math.sqrt( numColumns ) )
    shapeCols = math.ceil( math.sqrt( numColumns ) )
    i = 1
    for header in predictorsHeaders:
        plt.subplot( shapeRows, shapeCols, i )
        if logScalePredictors:
            plt.yscale("symlog")
        plt.scatter( predictors[header], labels, s=1, color='blue' )
        plt.xlabel( header )
        plt.ylabel( "Labels" )
        i += 1
    plt.subplot( shapeRows, shapeCols, i)
    plt.scatter( labels,  labels, s=1, color='blue' )
    plt.xlabel( "Labels" )
    plt.ylabel( "Labels" )
    #plt.savefig( "slopes.pdf" ) # if you wish to save the plot to a .pdf
    plt.tight_layout( )
    plt.show( )

def isCorrectSplit(dataFrameSize, trainDataSize, testDataSize, testRatio):
    isCorrect = True
    if trainDataSize/dataFrameSize != (1-testRatio):
        isCorrect =  False
    if testDataSize/dataFrameSize != testRatio:
        isCorrect =  False
    if isCorrect:
        print("Correct split.")
    else:
        print("Incorrect split.")
 
# print some basic information about the data 
def getInformation(dataFrame):
    print("First 5 rows:")
    print( dataFrame.head().to_string() )
    print()
    print("Column information:")
    dataFrame.info()
    print()
    print("Statisical description:")
    print( dataFrame.describe().to_string() )

def viewCorrelationToLabel(dataFrame, labelName=LABEL_NAME):
    corrMatrix = dataFrame.corr()
    correlations = corrMatrix[labelName].sort_values(ascending=False)
    print(correlations.to_string())

def viewScatterMatrix(dataFrame, headers=PREDICTORS_HEADERS): # use to see if any correlation between predictors exists
    scatter_matrix(dataFrame[headers]) 
    plt.show()

def preprocessData(dataFrame):
    processed = removeOutliers(dataFrame)
    return processed

def displayCVscores(pipeline, X, y, numCrosses=5): # given a non-fitted pipeline
    print("Cross Evaluation Resutls:")
    scores = cross_val_score(pipeline, X, y, scoring="neg_mean_squared_error", cv=numCrosses, n_jobs=-1)
    scores = np.sqrt(-scores)
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print()

def displayPredictionResults(pipeline, modelName, predictors, labels, predictionHeaders=PREDICTORS_HEADERS, displayModelResults=False): # given a fitted pipeline
    print("Prediction Results:")
    predictedLabels = pipeline.predict(predictors)
    MSE = mean_squared_error(labels, predictedLabels)
    RMSE = np.sqrt(MSE)
    if displayModelResults:
        headers = ["X", "Theta", "Theta Value"]
        table = PrettyTable(headers)
        model = pipeline.named_steps[modelName]
        for i in range(len(model.coef_)):
            x = "x_" + str(i)
            theta = "Theta_" + str(i)
            value = model.coef_[i]
            table.add_row([x, theta, value])
        print(table)
        print("Y-intercept:", model.intercept_)
    print("Mean Squared Error:", MSE)
    print("Root Mean Squared Error:", RMSE)
    print()

def createPipeline(degree=2, alpha=1.0, l1_ratio=0.5, tol=0.0001, max_iter=1000, threshold=None, modelName="ElasticNet"):
    if modelName == "ElasticNet":
        model = [ ("ElasticNet", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=tol, max_iter=max_iter)) ]
    elif modelName == "LinearRegression":
        model = [ ("FeatureRemoval", SelectFromModel(Lasso(alpha=alpha, tol=tol, max_iter=max_iter), threshold=threshold)), ("LinearRegression", LinearRegression(n_jobs=-1)) ]
    elif modelName == "Ridge":
        model = [ ("FeatureRemoval", SelectFromModel(Lasso(alpha=alpha, tol=tol, max_iter=max_iter), threshold=threshold)), ("Ridge", Ridge()) ]
    elif modelName == "Lasso":
        model = [ ("Lasso", Lasso()) ] 
    pipeline = Pipeline([
            ("PolynomialFeatures", PolynomialFeatures(degree=degree)),
            ("StandardScaler", StandardScaler()),
            ] + model )
    return pipeline

def performGridSearch(X, y, numCrosses=5, viewAllResults=False, modelName="ElasticNet"):
    pipeline = createPipeline(modelName=modelName)

    if modelName == "ElasticNet":
        paramGrid = {
            "PolynomialFeatures__degree" : [4],
            "ElasticNet__alpha" : [.005],
            "ElasticNet__l1_ratio" : [1],
            "ElasticNet__tol" : [.015],
            "ElasticNet__max_iter": [2500]
        }
    elif modelName == "LinearRegression":
        paramGrid = {
            "PolynomialFeatures__degree" : [4],
            "FeatureRemoval__threshold" : [2.5, 3, 3.5, None],
            "FeatureRemoval__estimator" : [Lasso(alpha=.005, tol=.015, max_iter=2500)]
        }
    elif modelName == "Ridge":
        paramGrid = {
            "PolynomialFeatures__degree" : [4],
            "FeatureRemoval__threshold" : [0,.5,1,2, None],
            "FeatureRemoval__estimator" : [Lasso(alpha=.005, tol=.015, max_iter=2500)]
        }
    gridSearch = GridSearchCV(pipeline, paramGrid, cv=numCrosses, scoring="neg_mean_squared_error", n_jobs=-1)
    gridSearch.fit(X, y)
    if viewAllResults:
        cvres = gridSearch.cv_results_
        for mean_score, std_deviation, params in zip(cvres["mean_test_score"], cvres['std_test_score'], cvres["params"]):
            print("RMSE:", np.sqrt(-mean_score), "Std:", std_deviation, "Params:", params)
    else:
        print("RMSE:", np.sqrt(-gridSearch.best_score_))
        print("Std:", gridSearch.cv_results_['std_test_score'][gridSearch.best_index_])
        print("Params:", gridSearch.best_params_)

def main():
    initialLoad = False
    performTrainTestSplit = False

    modelName = "ElasticNet"
    #modelName = "LinearRegression"
    #modelName = "Lasso"
    #modelName = "Ridge"
    searchGrid = True # perfrom a grid search to find the best pipeline params; evaluate must also be true
    evaluate = True

    finalTest = False # only perform this action once all evaluation on training data is complete

    if initialLoad:
        dataFrame = loadData()
    if performTrainTestSplit:
        testRatio = .2
        trainData, testData = trainTestSplit(dataFrame, testRatio)
        #trainData, testData = trainTestSplitStratified(dataFrame, "Age (days)", "Age Category", testRatio, 7, 120, 1.5)
        #isCorrectSplit(len(dataFrame), len(trainData), len(testData), testRatio)
        saveData(trainData, "train-data.txt")
        saveData(testData, "test-data.txt")

    if evaluate:
        trainDataRaw = loadData("train-data.txt")
        trainDataPreprocessed = preprocessData(trainDataRaw)
        #trainDataPreprocessed = trainDataRaw # without removing outliers
        X, y = predictorsLabelsSeparation(trainDataPreprocessed)
        if searchGrid:
            performGridSearch(X, y, numCrosses=11, viewAllResults=False, modelName=modelName) # search for best params to use
        else:
            pipeline = createPipeline(degree=4, alpha=0.005, l1_ratio=1, tol=.015, max_iter=2500, threshold=3, modelName=modelName)
            displayCVscores(pipeline, X, y, numCrosses=11)
            pipeline.fit(X, y)
            displayPredictionResults(pipeline, modelName, X, y, displayModelResults=False)


    if finalTest:
        print("***Final Test Results***")
        trainDataRaw = loadData("train-data.txt")
        testDataRaw = loadData("test-data.txt")
        trainDataPreprocessed = preprocessData(trainDataRaw)
        testDataPreprocessed = preprocessData(testDataRaw)
        Xtrain, ytrain = predictorsLabelsSeparation(trainDataPreprocessed)
        Xtest, ytest = predictorsLabelsSeparation(testDataPreprocessed)
        pipeline = createPipeline(degree=4, alpha=0.005, l1_ratio=1, tol=.015, max_iter=2500, threshold=3, modelName=modelName)
        pipeline.fit(Xtrain, ytrain)
        displayPredictionResults(pipeline, modelName, Xtest, ytest)


if __name__ == "__main__":
    main( )