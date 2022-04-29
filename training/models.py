import os
from scipy import stats
import joblib
import numpy as np
from sklearn import svm
from prepareData import noisyData, cleanData, data, preparenoisydata, preparecleandata
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

from my_constants import PREPROCESSED_DIR


def NaiveBayes(X, y):
    model = GaussianNB()
    return model


def Decision_Tree(X, y):
    scaler = StandardScaler()
    dt = DecisionTreeClassifier(random_state=0)
    model = Pipeline([('standardize', scaler),
                      ('dec_tree', dt)])
    return model


def SVM_classical(X, y):
    model = svm.SVC(C=1, kernel='rbf')
    # model.fit(X, y)
    return model


def SVM_CLASSWEIGHT_BALANCED(X, y):
    model = svm.SVC(C=1, class_weight='balanced', kernel='rbf')
    # model.fit(X, y)
    return model


def RandomForest(X, y):
    scaler = StandardScaler()
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    model = Pipeline([('standardize', scaler),
                    ('ran_for', rf)])
    #model.fit(X, y)
    return model

def Logistic(X, y):
    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=200)
    model = Pipeline([('standardize', scaler),
                    ('log_reg', lr)])
    #model.fit(X, y)
    return model

def Neighbor(X, y):
    scaler = StandardScaler()
    lr = KNeighborsClassifier(n_neighbors=2)
    model = Pipeline([('standardize', scaler),
                    ('log_reg', lr)])
    #model.fit(X, y)
    return model

def crossval(model, X, y, scoring_method):
    scores = cross_val_score(model, X, y, cv=10, scoring=scoring_method)
    return scores


def ttest(best_recall_scores, recallscores, labels):
    outcome={}
    for i, score in enumerate(recallscores):
        statistics, p_value = stats.ttest_rel(a=best_recall_scores,
                                          b=score
                                    )
        outcome[labels[i]]=p_value

    return outcome

def best_recall(data, labels, recallscores):
    best_recall = max(data["Recall"])
    idx_best_recall = data["Recall"].index(best_recall)
    recall_model_name = labels[idx_best_recall]
    print(recall_model_name)
    best_recall_scores = recallscores[idx_best_recall]

    return idx_best_recall, best_recall_scores, recall_model_name

def best_accuracy(data, labels, accuracyscores):
    best_accuracy = max(data["accuracy"])
    idx_best_accuracy = data["accuracy"].index(best_accuracy)
    accuracy_model_name = labels[idx_best_accuracy]
    print(accuracy_model_name)
    best_accuracy_scores = accuracyscores[idx_best_accuracy]

    return idx_best_accuracy, best_accuracy_scores, accuracy_model_name



if __name__ == "__main__":
    files = [f for f in os.listdir(PREPROCESSED_DIR) if '.csv' in f]
    recallscores=[]
    recallmeans=[]
    accuracyscores=[]
    accuracymeans=[]

    print("Testing classical model with noisy data")
    X_train_noisy, y_train_noisy = noisyData(files)
    classical_noisy= SVM_classical(X_train_noisy, y_train_noisy)
    classical_noisyrecall = crossval(classical_noisy, X_train_noisy, y_train_noisy, 'recall')
    classical_noisyaccuracy = crossval(classical_noisy, X_train_noisy, y_train_noisy, 'accuracy')
    classical_noisyprecision = crossval(classical_noisy, X_train_noisy, y_train_noisy, 'precision')
    classical_noisyF1 = crossval(classical_noisy, X_train_noisy, y_train_noisy, 'f1')
    
    print(classical_noisyrecall)
    print("recall: ", classical_noisyrecall.mean())
    recallscores.append(classical_noisyrecall)
    recallmeans.append(classical_noisyrecall.mean())

    print(classical_noisyaccuracy)
    print("accuracy: ", classical_noisyaccuracy.mean())
    accuracyscores.append(classical_noisyaccuracy)
    accuracymeans.append(classical_noisyaccuracy.mean())
    
    print("precision: ", classical_noisyprecision.mean())
    print("F1:", classical_noisyF1.mean())
    
    print("----------------------------------")



    print("Testing classical model with clean data")
    X_train_clean, y_train_clean = cleanData(files)
    classical_clean = SVM_classical(X_train_clean, y_train_clean)
    classical_cleanrecall = crossval(classical_clean, X_train_clean, y_train_clean, 'recall')
    classical_cleanaccuracy = crossval(classical_clean, X_train_clean, y_train_clean, 'accuracy')
    classical_cleanprecision = crossval(classical_clean, X_train_clean, y_train_clean, 'precision')
    classical_cleanF1 = crossval(classical_clean, X_train_clean, y_train_clean, 'f1')
    print(classical_cleanrecall)
    print("recall: ", classical_cleanrecall.mean())
    recallscores.append(classical_cleanrecall)
    recallmeans.append(classical_cleanrecall.mean())

    print(classical_cleanaccuracy)
    print("accuracy: ", classical_cleanaccuracy.mean())
    accuracyscores.append(classical_cleanaccuracy)
    accuracymeans.append(classical_cleanaccuracy.mean())
    
    print("precision: ", classical_cleanprecision.mean())
    print("F1: ", classical_noisyF1.mean())

    print("----------------------------------")



    print("Testing balanced model with noisy data")
    x_train, y_train, x_test, y_test= preparenoisydata(files)
    balanced_noisy = SVM_CLASSWEIGHT_BALANCED(X_train_noisy, y_train_noisy)
    balanced_noisyrecall = crossval(balanced_noisy, X_train_noisy, y_train_noisy, 'recall')
    balanced_noisyaccuracy = crossval(balanced_noisy, X_train_noisy, y_train_noisy, 'accuracy')
    balanced_noisyprecision = crossval(balanced_noisy, X_train_noisy, y_train_noisy, 'precision')
    balanced_noisyF1 = crossval(balanced_noisy, X_train_noisy, y_train_noisy, 'f1')
    print(balanced_noisyrecall)
    print("recall: ", balanced_noisyrecall.mean())
    recallscores.append(balanced_noisyrecall)
    recallmeans.append(balanced_noisyrecall.mean())
    print(balanced_noisyaccuracy)
    print("accuracy: ", balanced_noisyaccuracy.mean())
    accuracyscores.append(balanced_noisyaccuracy)
    accuracymeans.append(balanced_noisyaccuracy.mean())
    print("precision: ", balanced_noisyprecision.mean())
    print("F1: ", balanced_noisyF1.mean())

    
    
    
    print("----------------------------------")




    print("Testing balanced model with clean data")
    balanced_clean = SVM_CLASSWEIGHT_BALANCED(X_train_clean, y_train_clean)
    balanced_cleanrecall = crossval(balanced_clean, X_train_clean, y_train_clean, 'recall')
    balanced_cleanaccuracy = crossval(balanced_clean, X_train_clean, y_train_clean, 'accuracy')
    balanced_cleanprecision = crossval(balanced_clean, X_train_clean, y_train_clean, 'precision')
    balanced_cleanF1 = crossval(balanced_clean, X_train_clean, y_train_clean, 'f1')
    print(balanced_cleanrecall)
    print("recall: ", balanced_cleanrecall.mean())
    recallscores.append(balanced_cleanrecall)
    recallmeans.append(balanced_cleanrecall.mean())
    print(balanced_cleanaccuracy)
    print("accuracy: ", balanced_cleanaccuracy.mean())
    accuracyscores.append(balanced_cleanaccuracy)
    accuracymeans.append(balanced_cleanaccuracy.mean())
    print("precision: ", balanced_cleanprecision.mean())
    print("F1: ", balanced_cleanF1.mean())


    print("----------------------------------")

    print("Testing Random forest model with noisy data")
    Random_noisy = RandomForest(X_train_noisy, y_train_noisy)

    Random_noisyrecall = crossval(Random_noisy, X_train_noisy, y_train_noisy, 'recall')
    Random_noisyaccuracy = crossval(Random_noisy, X_train_noisy, y_train_noisy, 'accuracy')
    Random_noisyprecision = crossval(Random_noisy, X_train_noisy, y_train_noisy, 'precision')
    Random_noisyF1 = crossval(Random_noisy, X_train_noisy, y_train_noisy, 'f1')
    print(Random_noisyrecall)
    print("recall: ", Random_noisyrecall.mean())
    recallscores.append(Random_noisyrecall)
    recallmeans.append(Random_noisyrecall.mean())

    print(Random_noisyaccuracy)
    print("accuracy: ", Random_noisyaccuracy.mean())
    accuracyscores.append(Random_noisyaccuracy)
    accuracymeans.append(Random_noisyaccuracy.mean())
    print("precision: ", Random_noisyprecision.mean())
    print("F1: ", Random_noisyF1.mean())


    print("----------------------------------")

    print("Testing Random forest model with clean data")
    #x_train, y_train, x_test, y_test = preparedata(files)
    Random_clean = RandomForest(X_train_clean, y_train_clean)

    Random_cleanrecall = crossval(Random_clean, X_train_clean, y_train_clean, 'recall')
    Random_cleanaccuracy = crossval(Random_clean, X_train_clean, y_train_clean, 'accuracy')
    Random_cleanprecision = crossval(Random_clean, X_train_clean, y_train_clean, 'precision')
    Random_cleanF1 = crossval(Random_clean, X_train_clean, y_train_clean, 'f1')
    print(Random_cleanrecall)
    print("recall: ", Random_cleanrecall.mean())
    recallscores.append(Random_cleanrecall)
    recallmeans.append(Random_cleanrecall.mean())

    print(Random_cleanaccuracy)
    print("accuracy: ", Random_cleanaccuracy.mean())
    accuracyscores.append(Random_cleanaccuracy)
    accuracymeans.append(Random_cleanaccuracy.mean())
    print("precision: ", Random_cleanprecision.mean())
    print("F1: ", Random_cleanF1.mean())


    print("----------------------------------")


    print("Testing Naive Bayes model with noisy data")
    NaiveBayes_noisy = NaiveBayes(X_train_noisy, y_train_noisy)
    NaiveBayes_noisyrecall = crossval(NaiveBayes_noisy, X_train_noisy, y_train_noisy, 'recall')
    NaiveBayes_noisyaccuracy = crossval(NaiveBayes_noisy, X_train_noisy, y_train_noisy, 'accuracy')
    NaiveBayes_noisyprecision = crossval(NaiveBayes_noisy, X_train_noisy, y_train_noisy, 'precision')
    NaiveBayes_noisyF1 = crossval(NaiveBayes_noisy, X_train_noisy, y_train_noisy, 'f1')
    print(NaiveBayes_noisyrecall)
    print("recall: ", NaiveBayes_noisyrecall.mean())
    recallscores.append(NaiveBayes_noisyrecall)
    recallmeans.append(NaiveBayes_noisyrecall.mean())

    print(NaiveBayes_noisyaccuracy)
    print("accuracy: ", NaiveBayes_noisyaccuracy.mean())
    accuracyscores.append(NaiveBayes_noisyaccuracy)
    accuracymeans.append(NaiveBayes_noisyaccuracy.mean())
    print("precision: ", NaiveBayes_noisyprecision.mean())
    print("F1: ", NaiveBayes_noisyF1.mean())


    print("----------------------------------")

    print("Testing Naive Bayes model with clean data")
    NaiveBayes_clean = NaiveBayes(X_train_clean, y_train_clean)
    NaiveBayes_cleanrecall = crossval(NaiveBayes_clean, X_train_clean, y_train_clean, 'recall')
    NaiveBayes_cleanaccuracy = crossval(NaiveBayes_clean, X_train_clean, y_train_clean, 'accuracy')
    NaiveBayes_cleanprecision = crossval(NaiveBayes_clean, X_train_clean, y_train_clean, 'precision')
    NaiveBayes_cleanF1 = crossval(NaiveBayes_clean, X_train_clean, y_train_clean, 'f1')
    print(NaiveBayes_cleanrecall)
    print("recall: ", NaiveBayes_cleanrecall.mean())
    recallscores.append(NaiveBayes_cleanrecall)
    recallmeans.append(NaiveBayes_cleanrecall.mean())

    print(NaiveBayes_cleanaccuracy)
    print("accuracy: ", NaiveBayes_cleanaccuracy.mean())
    accuracyscores.append(NaiveBayes_cleanaccuracy)
    accuracymeans.append(NaiveBayes_cleanaccuracy.mean())
    print("precision: ", NaiveBayes_cleanprecision.mean())
    print("F1: ", NaiveBayes_cleanF1.mean())


    print("----------------------------------")



    print("Testing Decision tree model with noisy data")
    Decision_Tree_noisy = Decision_Tree(X_train_noisy, y_train_noisy)
    Decision_Tree_noisyrecall = crossval(Decision_Tree_noisy, X_train_noisy, y_train_noisy, 'recall')
    Decision_Tree_noisyaccuracy = crossval(Decision_Tree_noisy, X_train_noisy, y_train_noisy, 'accuracy')
    Decision_Tree_noisyprecision = crossval(Decision_Tree_noisy, X_train_noisy, y_train_noisy, 'precision')
    Decision_Tree_noisyF1 = crossval(Decision_Tree_noisy, X_train_noisy, y_train_noisy, 'f1')
    print(Decision_Tree_noisyrecall)
    print("recall: ", Decision_Tree_noisyrecall.mean())
    recallscores.append(Decision_Tree_noisyrecall)
    recallmeans.append(Decision_Tree_noisyrecall.mean())

    print(Decision_Tree_noisyaccuracy)
    print("accuracy: ", Decision_Tree_noisyaccuracy.mean())
    accuracyscores.append(Decision_Tree_noisyaccuracy)
    accuracymeans.append(Decision_Tree_noisyaccuracy.mean())
    print("precision: ", Decision_Tree_noisyprecision.mean())
    print("F1: ", Decision_Tree_noisyF1.mean())


    print("----------------------------------")

    print("Testing Decision tree model with clean data")
    Decision_Tree_clean = Decision_Tree(X_train_clean, y_train_clean)
    Decision_Tree_cleanrecall = crossval(Decision_Tree_clean, X_train_clean, y_train_clean, 'recall')
    Decision_Tree_cleanaccuracy = crossval(Decision_Tree_clean, X_train_clean, y_train_clean, 'accuracy')
    Decision_Tree_cleanprecision = crossval(Decision_Tree_clean, X_train_clean, y_train_clean, 'precision')
    Decision_Tree_cleanF1 = crossval(Decision_Tree_clean, X_train_clean, y_train_clean, 'f1')
    print(Decision_Tree_cleanrecall)
    print("recall: ", Decision_Tree_cleanrecall.mean())
    recallscores.append(Decision_Tree_cleanrecall)
    recallmeans.append(Decision_Tree_cleanrecall.mean())

    print(Decision_Tree_cleanaccuracy)
    print("accuracy: ", Decision_Tree_cleanaccuracy.mean())
    accuracyscores.append(Decision_Tree_cleanaccuracy)
    accuracymeans.append(Decision_Tree_cleanaccuracy.mean())
    print("precision: ", Decision_Tree_cleanprecision.mean())
    print("F1: ", Decision_Tree_cleanF1.mean())



    print("----------------------------------")

    print("Testing Logistic Regression model with noisy data")
    #x_train, x_test, y_train, y_test = data(files)

    Logistic_noisy = Logistic(X_train_noisy, y_train_noisy)

    Logistic_noisyrecall = crossval(Logistic_noisy, X_train_noisy, y_train_noisy, 'recall')
    Logistic_noisyaccuracy = crossval(Logistic_noisy, X_train_noisy, y_train_noisy, 'accuracy')
    Logistic_noisyprecision = crossval(Logistic_noisy, X_train_noisy, y_train_noisy, 'precision')
    Logistic_noisyF1 = crossval(Logistic_noisy, X_train_noisy, y_train_noisy, 'f1')
    print(Logistic_noisyrecall)
    print("recall: ", Logistic_noisyrecall.mean())
    recallscores.append(Logistic_noisyrecall)
    recallmeans.append(Logistic_noisyrecall.mean())

    print(Logistic_noisyaccuracy)
    print("accuracy: ", Logistic_noisyaccuracy.mean())
    accuracyscores.append(Logistic_noisyaccuracy)
    accuracymeans.append(Logistic_noisyaccuracy.mean())
    print("precision: ", Logistic_noisyprecision.mean())
    print("F1: ", Logistic_noisyF1.mean())

    
    print("----------------------------------")


    print("Testing Logistic Regression model with clean data")
    # x_train, x_test, y_train, y_test = data(files)

    Logistic_clean = Logistic(X_train_clean, y_train_clean)

    Logistic_cleanrecall = crossval(Logistic_clean, X_train_clean, y_train_clean, 'recall')
    Logistic_cleanaccuracy = crossval(Logistic_clean, X_train_clean, y_train_clean, 'accuracy')
    Logistic_cleanprecision = crossval(Logistic_clean, X_train_clean, y_train_clean, 'precision')
    Logistic_cleanF1 = crossval(Logistic_clean, X_train_clean, y_train_clean, 'f1')
    print(Logistic_cleanrecall)
    print("recall: ", Logistic_cleanrecall.mean())
    recallscores.append(Logistic_cleanrecall)
    recallmeans.append(Logistic_cleanrecall.mean())

    print(Logistic_cleanaccuracy)
    print("accuracy: ", Logistic_cleanaccuracy.mean())
    accuracyscores.append(Logistic_cleanaccuracy)
    accuracymeans.append(Logistic_cleanaccuracy.mean())
    print("precision: ", Logistic_cleanprecision.mean())
    print("F1: ", Logistic_cleanF1.mean())


    print("----------------------------------")


    print("Testing KNN model with noisy data")
    KNN_noisy = Neighbor(X_train_noisy, y_train_noisy)

    KNN_noisyrecall = crossval(KNN_noisy, X_train_noisy, y_train_noisy, 'recall')
    KNN_noisyaccuracy = crossval(KNN_noisy, X_train_noisy, y_train_noisy, 'accuracy')
    KNN_noisyprecision = crossval(KNN_noisy, X_train_noisy, y_train_noisy, 'precision')
    KNN_noisyF1 = crossval(KNN_noisy, X_train_noisy, y_train_noisy, 'f1')
    print(KNN_noisyrecall)
    print("recall: ", KNN_noisyrecall.mean())
    recallscores.append(KNN_noisyrecall)
    recallmeans.append(KNN_noisyrecall.mean())

    print(KNN_noisyaccuracy)
    print("accuracy: ", KNN_noisyaccuracy.mean())
    accuracyscores.append(KNN_noisyaccuracy)
    accuracymeans.append(KNN_noisyaccuracy.mean())
    print("precision: ", KNN_noisyprecision.mean())
    print("F1: ", KNN_noisyF1.mean())

    print("----------------------------------")

    print("Testing KNN model with clean data")
    KNN_clean = Neighbor(X_train_clean, y_train_clean)

    KNN_cleanrecall = crossval(KNN_clean, X_train_clean, y_train_clean, 'recall')
    KNN_cleanaccuracy = crossval(KNN_clean, X_train_clean, y_train_clean, 'accuracy')
    KNN_cleanprecision = crossval(KNN_clean, X_train_clean, y_train_clean, 'precision')
    KNN_cleanF1 = crossval(KNN_clean, X_train_clean, y_train_clean, 'f1')
    print(KNN_cleanrecall)
    print("recall: ", KNN_cleanrecall.mean())
    recallscores.append(KNN_cleanrecall)
    recallmeans.append(KNN_cleanrecall.mean())

    print(KNN_cleanaccuracy)
    print("accuracy: ", KNN_cleanaccuracy.mean())
    accuracyscores.append(KNN_cleanaccuracy)
    accuracymeans.append(KNN_cleanaccuracy.mean())
    print("precision: ", KNN_cleanprecision.mean())
    print("F1: ", KNN_cleanF1.mean())

    print("----------------------------------")



    data = {
        'Model': ['classical_noisy', 'classical_clean', 'balanced_noisy', 'balanced_clean', 'Random_noisy', 'Random_clean', 'NaiveBayes_noisy','NaiveBayes_clean', 'Decision_Tree_noisy','Decision_Tree_clean','Logistic_noisy', 'Logistic_clean', 'KNN_noisy', 'KNN_clean' ],
        'Recall': [classical_noisyrecall.mean(), classical_cleanrecall.mean(), balanced_noisyrecall.mean(), balanced_cleanrecall.mean(), Random_noisyrecall.mean(), Random_cleanrecall.mean(), NaiveBayes_noisyrecall.mean(), NaiveBayes_cleanrecall.mean(), Decision_Tree_noisyrecall.mean(), Decision_Tree_cleanrecall.mean(), Logistic_noisyrecall.mean(), Logistic_cleanrecall.mean(), KNN_noisyrecall.mean(), KNN_cleanrecall.mean() ],
        'accuracy': [classical_noisyaccuracy.mean(), classical_cleanaccuracy.mean(), balanced_noisyaccuracy.mean(),
                   balanced_cleanaccuracy.mean(), Random_noisyaccuracy.mean(), Random_cleanaccuracy.mean(),
                   NaiveBayes_noisyaccuracy.mean(), NaiveBayes_cleanaccuracy.mean(), Decision_Tree_noisyaccuracy.mean(),
                   Decision_Tree_cleanaccuracy.mean(), Logistic_noisyaccuracy.mean(), Logistic_cleanaccuracy.mean(),
                   KNN_noisyaccuracy.mean(), KNN_cleanaccuracy.mean()]

    }
    labels = ['classical_noisy', 'classical_clean', 'balanced_noisy', 'balanced_clean', 'Random_noisy', 'Random_clean',
              'NaiveBayes_noisy', 'NaiveBayes_clean', 'Decision_Tree_noisy', 'Decision_Tree_clean', 'Logistic_noisy',
              'Logistic_clean', 'KNN_noisy', 'KNN_clean']

    idx_best_recall, best_recall_scores, recall_model_name= best_recall(data, labels, recallscores)
    print(f'Model Recall scores {recallscores[idx_best_recall]}\nRecall Mean: {recallscores[idx_best_recall].mean()}')
    recalloutcome = ttest(best_recall_scores, recallscores, labels)
    print(recalloutcome)

    idx_best_accuracy, best_accuracy_scores, accuracy_model_name= best_accuracy(data, labels, accuracyscores)
    print(f'Model accuracy scores {accuracyscores[idx_best_accuracy]}\naccuracyMean: {accuracyscores[idx_best_accuracy].mean()}')
    accuracyoutcome = ttest(best_accuracy_scores, accuracyscores, labels)
    print(accuracyoutcome)

    if data["Recall"][idx_best_accuracy] >=90:
        best_model=labels[idx_best_accuracy]
    else:
        data["Model"].pop(idx_best_accuracy)
        data["accuracy"].pop(idx_best_accuracy)
        data["Recall"].pop(idx_best_accuracy)
        accuracyscores.pop(idx_best_accuracy)
        recallscores.pop(idx_best_accuracy)
        accuracymeans.pop(idx_best_accuracy)
        recallmeans.pop(idx_best_accuracy)
        idx_best_recall, best_recall_scores, recall_model_name = best_recall(data, labels, recallscores)
        print(
            f'Model Recall scores {recallscores[idx_best_recall]}\nRecall Mean: {recallscores[idx_best_recall].mean()}')
        recalloutcome = ttest(best_recall_scores, recallscores, labels)
        print(recalloutcome)

        idx_best_accuracy, best_accuracy_scores, accuracy_model_name = best_accuracy(data, labels, accuracyscores)
        print(
            f'Model accuracy scores {accuracyscores[idx_best_accuracy]}\naccuracyMean: {accuracyscores[idx_best_accuracy].mean()}')
        accuracyoutcome = ttest(best_accuracy_scores, accuracyscores, labels)
        print(accuracyoutcome)
        best_model=labels[idx_best_accuracy]











