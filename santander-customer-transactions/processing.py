import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import math, os

LABEL_NAME = "target"
TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"
PRIMARY_KEY = "ID_code"
TRAIN_GROUPS_PATH = "data/train-groups/"
MODELS_PATH = "models/"
SINGLE_TRAIN_GROUP_PATH = "data/train-groups/train_group_0.csv"
SUBMISSION_PATH = "submission.csv"

def fetchData(filePath=TRAIN_DATA_PATH, primaryKey=PRIMARY_KEY):
    return pd.read_csv(filePath, index_col=primaryKey)

def separatePredictorsLabels(dataFrame, labelName=LABEL_NAME):
    predictors = dataFrame.drop(labelName, axis=1)
    labels = dataFrame[labelName].copy()
    return predictors, labels

def separateYesNoLabelGroups(dataFrame, labelName=LABEL_NAME): 
    yesGroup = dataFrame[dataFrame[labelName] == 1]
    noGroup = dataFrame[dataFrame[labelName] == 0]
    numNoGroups = math.ceil(noGroup.shape[0]/yesGroup.shape[0])
    numNosPerGroup = math.floor(noGroup.shape[0]/numNoGroups)
    noGroups = []
    numNosLeftOver = noGroup.shape[0]-(numNosPerGroup*numNoGroups)
    for i in range(numNoGroups):
        if i == 0:
            subGroup = noGroup.sample(numNosPerGroup+numNosLeftOver, random_state=42)
            noGroups.append(subGroup)
        else:
            subGroup = noGroup.sample(numNosPerGroup, random_state=42)
            noGroups.append(subGroup)
        noGroup = noGroup.drop(subGroup.index)
    return yesGroup, noGroups

def mergeYesGroupWithNoGroups(yesGroup, noGroups):
    groups = []
    for noGroup in noGroups:
        mergedGroup = yesGroup.append(noGroup).sample(frac=1, random_state=42) 
        groups.append(mergedGroup)
    return groups

def saveTrainGroups(trainGroups, savePath=TRAIN_GROUPS_PATH):
    for i in range(len(trainGroups)):
        fileName = "train_group_" + str(i) + ".csv"
        trainGroups[i].to_csv(savePath+fileName)

def fetchSingleTrainGroup(trainGroupPath=SINGLE_TRAIN_GROUP_PATH, primaryKey=PRIMARY_KEY):
    return pd.read_csv(trainGroupPath, index_col=primaryKey)

def crossEvaluateModel(model, predictors, labels, numCrosses=5):
    yscores = cross_val_predict(model, predictors, labels, cv=numCrosses, method="predict_proba", n_jobs=-1, verbose=3)
    score = roc_auc_score(labels, yscores[:,1])
    print("ROC AUC Score:", score)
    scores = cross_val_score(model, predictors, labels, cv=numCrosses, scoring="accuracy", n_jobs=-1)
    print("Accuracy scores:")
    print(scores)
    predictedLabels = cross_val_predict(model, predictors, labels, cv=numCrosses, n_jobs=-1)
    print("Confusion matrix:")
    print( confusion_matrix(labels, predictedLabels) )
    print("Recall Score:", recall_score(labels, predictedLabels, average="micro"))
    print("Precision Score:", precision_score(labels, predictedLabels, average="micro"))

def createPipeline(modelName="RandomForestClassifier"):
    if modelName == "RandomForestClassifier":
        model = [ ("Model", RandomForestClassifier(n_estimators=300, bootstrap=False, max_features="sqrt", criterion="entropy", random_state=42, n_jobs=-1)) ]
    if modelName == "AdaBoostClassifier":
        model = [ ("Model", AdaBoostClassifier(DecisionTreeClassifier(random_state=42, criterion="entropy"), n_estimators=100, random_state=42)) ]
    pipeline = Pipeline([
        ("scale_features", StandardScaler())
        ] + model )
    return pipeline

def performGridSearch(X, y, numCrosses=5, viewAllResults=False, modelName="RandomForestClassifier"):
    pipeline = createPipeline(modelName=modelName)
    if modelName == "RandomForestClassifier":
        paramGrid = {
            #"Model__criterion": ["gini", "entropy"],
            "Model__criterion": ["entropy"],
            "Model__n_estimators": [10],
            "Model__max_features": ["sqrt", None],
            #"Model__bootstrap": [True, False]
        }
    gridSearch = GridSearchCV(pipeline, paramGrid, cv=numCrosses, scoring="roc_auc", n_jobs=-1, verbose=3)
    gridSearch.fit(X, y)
    if viewAllResults:
        cvres = gridSearch.cv_results_
        for mean_score, std_deviation, params in zip(cvres["mean_test_score"], cvres['std_test_score'], cvres["params"]):
            print("Mean ROC AUC Score:", mean_score, "Std:", std_deviation, "Params:", params)
    else:
        print("Best score:", gridSearch.best_score_)
        print("Params:", gridSearch.best_params_)

def generateTrainedModels(trainGroupsPath=TRAIN_GROUPS_PATH, modelsSavePath=MODELS_PATH, primaryKey=PRIMARY_KEY):
    for fileName in os.listdir(trainGroupsPath): 
        trainFilePath = os.path.join(trainGroupsPath, fileName)
        trainDataFrame = pd.read_csv(trainFilePath, index_col=primaryKey)
        trainXraw, trainYraw = separatePredictorsLabels(trainDataFrame)
        pipeline = createPipeline()
        pipeline.fit(trainXraw, trainYraw)
        modelFileName = fileName.split('.')[0] + "_model.joblib"
        modelPath = os.path.join(modelsSavePath, modelFileName)
        saveModel(pipeline, modelPath)

def fetchTrainedModels(modelsPath=MODELS_PATH):
    models = []
    for fileName in os.listdir(modelsPath): 
        modelPath = os.path.join(modelsPath, fileName)
        model = loadModel(modelPath)
        models.append(model)
    return models

def getPredictions(dataFrame):
    predictions = []
    predictionProbs = []
    models = fetchTrainedModels()
    for model in models:
        predictionProbs.append(model.predict_proba(dataFrame))
    for i in range(predictionProbs[0].shape[0]):
        votingRes = {"probZero": 0, "probOne": 0 }
        for j in range(len(predictionProbs)):
            prob = predictionProbs[j][i]
            votingRes["probZero"] += prob[0]
            votingRes["probOne"] += prob[1]
            votingRes["probZero"] /= len(predictionProbs)
            votingRes["probOne"] /= len(predictionProbs)
        if votingRes["probOne"] > votingRes["probZero"]:
            predictions.append(1)
        else:
            predictions.append(0)
    return np.array(predictions)

def saveModel(model, modelPath):
    joblib.dump(model, modelPath) 
def loadModel(modelPath):
    return joblib.load(modelPath)

def generateSubmissionFile(testDataPath=TEST_DATA_PATH, submissionPath=SUBMISSION_PATH):
    testDataFrame = fetchData(testDataPath)
    predictions = getPredictions(testDataFrame)
    submission = pd.DataFrame( data=predictions, columns=["target"], index=testDataFrame.index )
    submission.to_csv(submissionPath)
    

def main():
    prepareTrainGroups = False
    evaluateOnSingleTrainGroup = True
    searchGridOnSingleTrainGroup = False

    generateModels = False
    readyToSubmit = False

    if prepareTrainGroups:
        dataFrame = fetchData()
        yesGroup, noGroups = separateYesNoLabelGroups(dataFrame)
        trainGroups = mergeYesGroupWithNoGroups(yesGroup, noGroups)
        saveTrainGroups(trainGroups)

    if evaluateOnSingleTrainGroup:
        trainDataFrame = fetchSingleTrainGroup()
        trainXraw, trainYraw = separatePredictorsLabels(trainDataFrame)
        #pipeline = createPipeline(modelName="AdaBoostClassifier")
        pipeline = createPipeline(modelName="RandomForestClassifier")
        crossEvaluateModel(pipeline, trainXraw, trainYraw, 4)
    
    if searchGridOnSingleTrainGroup:
        trainDataFrame = fetchSingleTrainGroup()
        trainXraw, trainYraw = separatePredictorsLabels(trainDataFrame)
        performGridSearch(trainXraw, trainYraw, viewAllResults=False)

    if generateModels:
        generateTrainedModels()

    if readyToSubmit:
        generateSubmissionFile()
    
if __name__ == "__main__":
    main()
