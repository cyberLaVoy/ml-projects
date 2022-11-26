#!/usr/bin/env python3
def warn(*args, **kwargs): # don't display any warnings
    pass
import warnings
warnings.warn = warn

import joblib, cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_PATH = "RubiksCube-train-01.csv"
MODEL_FILE_NAME = "model.joblib"
LABEL_NAME = "level"
TRAIN_DATA_PATH = "train-rubiks-cube_data.csv"
TEST_DATA_PATH = "test-rubiks-cube_data.csv"

# Final exam predictions save path
FINAL_TEST_DATA_PATH = "test-rubiks-cube_data.csv" # change this to whatever the final testing data file name is
FINAL_PREDICTIONS_SAVE_PATH = "final-testing-predictions.csv" # change this ONLY IF submission file name needs to be specific

def fetchData(dataPath=DATA_PATH):
    return pd.read_csv(dataPath)
def saveData(data, savePath):
    data.to_csv(savePath, index=False)

def fetchAndPreprocessData():
    dataSet = fetchData()
    dataSet = duplicateUnderrepresentedRows(dataSet)
    return shuffleData(dataSet)

def shuffleData(dataFrame):
    return dataFrame.sample(frac=1, random_state=42).reset_index(drop=True)

def separatePredictorsAndLabels(dataFrame, labelName=LABEL_NAME):
    predictors = dataFrame.drop(labelName, axis=1)
    labels = dataFrame[labelName].copy()
    return predictors, labels

def trainTestSplit(dataFrame, testRatio):
    return train_test_split(dataFrame, test_size=testRatio, random_state=42)


def createPipline():
    model = [ ("model", RandomForestClassifier(n_estimators=75, criterion="entropy", random_state=42, n_jobs=-1)) ]
    #model = [ ("model", SGDClassifier(random_state=42, n_jobs=-1)) ]
    #model = [ ("model", AdaBoostClassifier(DecisionTreeClassifier(criterion="entropy"), n_estimators=50, random_state=42)) ]
    pipeline = Pipeline([
        ("scale_features", StandardScaler())
        ] + model )
    return pipeline

def saveModel(model, modelFileName=MODEL_FILE_NAME):
    joblib.dump(model, modelFileName) 
def loadModel(modelFileName=MODEL_FILE_NAME):
    return joblib.load(modelFileName)

def fitAndSaveModel(model, trainData):
    Xtrain, ytrain = separatePredictorsAndLabels(trainData)
    model.fit(Xtrain, ytrain)
    saveModel(model)

def crossEvaluateModel(model, predictors, labels, numCrosses=11):
    scores = cross_val_score(model, predictors, labels, cv=numCrosses, scoring="accuracy")
    print("Accuracy scores:")
    print(scores)
    predictedLabels = cross_val_predict(model, predictors, labels, cv=numCrosses)
    print("Confusion matrix:")
    print( confusion_matrix(labels, predictedLabels) )
    print("Recall Score:", recall_score(labels, predictedLabels, average="micro"))
    print("Precision Score:", precision_score(labels, predictedLabels, average="micro"))

def testModel(model, predictors, labels):
    predictedLabels = model.predict(predictors)
    print("Accuracy score:", accuracy_score(labels, predictedLabels))
    print("Confusion matrix:")
    print( confusion_matrix(labels, predictedLabels) )
    print("Recall Score:", recall_score(labels, predictedLabels, average="micro"))
    print("Precision Score:", precision_score(labels, predictedLabels, average="micro"))

def displayLabelCounts(dataFrame, labelName=LABEL_NAME):
    print(dataFrame[labelName].value_counts())
    print("Maximum count:", max(dataFrame[labelName].value_counts()))

def duplicateUnderrepresentedRows(dataFrame, labelName=LABEL_NAME):
    maximumCount = max(dataFrame[labelName].value_counts())
    for uniqueLevel in dataFrame[labelName].unique():
        duplicationAmount = maximumCount - (dataFrame[labelName] == uniqueLevel).sum()
        duplicationRows = dataFrame[ (dataFrame[labelName] == uniqueLevel) ]
        # duplicate, with replacement, random rows
        dataFrame = dataFrame.append(duplicationRows.sample(duplicationAmount, replace=True, random_state=42)) 
    return dataFrame

def saveTestPredictions(model, features, savePath=FINAL_PREDICTIONS_SAVE_PATH):
    predictions = model.predict(features)
    predictions = pd.DataFrame(predictions, columns=["level"])
    saveData(features.join(predictions), savePath)


def main():
    saveDataSplit = False
    evaluate = False
    saveTrainedModel = False
    loadAndTest = False

    # Final exam variables
    saveFinalExamModel = False # make sure this step is performed, before the final exam
    savePredictions = True

    if saveDataSplit:
        dataSet = fetchAndPreprocessData()
        trainData, testData = trainTestSplit(dataSet, .2)
        saveData(trainData, TRAIN_DATA_PATH)
        saveData(testData, TEST_DATA_PATH)
    if evaluate:
        pipeline = createPipline()
        trainData = fetchData(TRAIN_DATA_PATH)
        Xtrain, ytrain = separatePredictorsAndLabels(trainData)
        crossEvaluateModel(pipeline, Xtrain, ytrain)
    if saveTrainedModel:
        pipeline = createPipline()
        trainData = fetchData(TRAIN_DATA_PATH)
        fitAndSaveModel(pipeline, trainData)
    if loadAndTest:
        model = loadModel()
        testData = fetchData(TEST_DATA_PATH)
        Xtest, ytest = separatePredictorsAndLabels(testData)
        testModel(model, Xtest, ytest)

    # Final exam steps
    if saveFinalExamModel:
        pipeline = createPipline()
        trainData = fetchAndPreprocessData()
        fitAndSaveModel(pipeline, trainData)
    if savePredictions:
        model = loadModel()
        testData = fetchData(FINAL_TEST_DATA_PATH)
        if "level" in list(testData): # if labels are present
            Xtest, ytest = separatePredictorsAndLabels(testData) # drop labels
        else: # test data contains only features
            Xtest = testData
        saveTestPredictions(model, Xtest)

if __name__ == "__main__":
    main()