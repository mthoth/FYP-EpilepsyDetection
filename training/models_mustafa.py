from my_constants import PREPROCESSED_DIR
import os
import joblib

import numpy as np
import pandas as pd
from sklearn import svm
from prepareData import noisyData, cleanData
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

def SVM_classical(X, y):
    model = svm.SVC(C=1, kernel='rbf')
    #model.fit(X, y)
    return model


def SVM_CLASSWEIGHT_BALANCED(X, y):
    model = svm.SVC(C=1, class_weight='balanced', kernel='rbf')
    #model.fit(X, y)
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
    kn = KNeighborsClassifier(n_neighbors=2)
    model = Pipeline([('standardize', scaler),
                    ('k_near', kn)])
    #model.fit(X, y)
    return model

def NaiveBayes(X, y):
    model = GaussianNB()
    return model


def Decision_Tree(X, y):
    scaler = StandardScaler()
    dt = DecisionTreeClassifier(random_state=0)
    model = Pipeline([('standardize', scaler),
                    ('dec_tree', dt)])
    return model


def crossval(model, X, y):
    scores = cross_val_score(model, X, y, cv=10, scoring="recall")
    return scores


def best_model(models):

    return 0


if __name__ == "__main__":

    files = [f for f in os.listdir(PREPROCESSED_DIR) if '.csv' in f]
    print("Testing classical model with noisy data")
    X_train_noisy, y_train_noisy, X_test_noisy, y_test_noisy = noisyData(files)
    model1 = SVM_classical(X_train_noisy, y_train_noisy)
    model1scores = crossval(model1, X_train_noisy, y_train_noisy)
    model1.fit(X_train_noisy, y_train_noisy)
    print(model1scores)
    print("average: ", model1scores.mean())

    model1_predictions = model1.predict(X_test_noisy)
    model1_classification_report = classification_report(y_test_noisy, model1_predictions, output_dict=True)
    model1_confusion_matrix = confusion_matrix(y_test_noisy, model1_predictions)
    #print(model1_classification_report)
    #print(model1_confusion_matrix)

    print("Testing classical model with clean data")
    X_train_clean, y_train_clean, X_test_clean, y_test_clean = cleanData(files)
    model2 = SVM_classical(X_train_clean, y_train_clean)
    model2scores = crossval(model2, X_train_noisy, y_train_noisy)
    model2.fit(X_train_clean, y_train_clean)
    print(model2scores)
    print("average: ", model2scores.mean())

    model2_predictions = model2.predict(X_test_clean)
    model2_classification_report = classification_report(y_test_clean, model2_predictions, output_dict=True)
    model2_confusion_matrix = confusion_matrix(y_test_clean, model2_predictions)
    #print(model2_classification_report)
    #print(model2_confusion_matrix)

    print("Testing balanced model with noisy data")
    # X_train_noisy, y_train_noisy, X_test_noisy, y_test_noisy = noisyData(files)
    model3 = SVM_CLASSWEIGHT_BALANCED(X_train_noisy, y_train_noisy)
    model3scores = crossval(model3, X_train_noisy, y_train_noisy)
    model3.fit(X_train_noisy, y_train_noisy)
    print(model3scores)
    print("average: ", model3scores.mean())

    model3_predictions = model3.predict(X_test_noisy)
    model3_classification_report = classification_report(y_test_noisy, model3_predictions, output_dict=True)
    model3_confusion_matrix = confusion_matrix(y_test_noisy, model3_predictions)
    
    joblib.dump(model3, 'chb20model.pkl')
    
    # print(model3_classification_report)
    # print(model3_confusion_matrix)

    print("Testing balanced model with clean data")
    # X_train_clean, y_train_clean, X_test_clean, y_test_clean = cleanData(files)
    model4 = SVM_CLASSWEIGHT_BALANCED(X_train_clean, y_train_clean)
    model4scores = crossval(model4, X_test_clean, y_test_clean)
    model4.fit(X_train_clean, y_train_clean)
    print(model4scores)
    print("average: ", model4scores.mean())
    model4_predictions = model4.predict(X_test_clean)
    model4_classification_report = classification_report(y_test_clean, model4_predictions, output_dict=True)
    model4_confusion_matrix = confusion_matrix(y_test_clean, model4_predictions)
    # print(model4_classification_report)
    # print(model4_confusion_matrix)

    print("Testing Random forest model with noisy data")
    #X_train_noisy, y_train_noisy, X_test_noisy, y_test_noisy = noisyData(files)
    model5 = RandomForest(X_train_noisy, y_train_noisy)
    model5scores = crossval(model5, X_test_noisy, y_test_noisy)
    model5.fit(X_train_noisy, y_train_noisy)
    print(model5scores)
    print("average: ", model5scores.mean())

    model5_predictions = model5.predict(X_test_noisy)
    model5_classification_report = classification_report(y_test_noisy, model5_predictions, output_dict=True)
    model5_confusion_matrix = confusion_matrix(y_test_noisy, model5_predictions)
    # print(model5_classification_report)
    # print(model5_confusion_matrix)

    

    print("Testing Naive Bayes model with noisy data")
    model6 = NaiveBayes(X_train_noisy, y_train_noisy)
    model6scores = crossval(model6, X_train_noisy, y_train_noisy)
    print(model6scores)
    print("average: ", model6scores.mean())

    model6.fit(X_train_noisy, y_train_noisy)
    model6_predictions = model6.predict(X_test_noisy)
    model6_classification_report = classification_report(y_test_noisy, model6_predictions, output_dict=True)
    model6_confusion_matrix = confusion_matrix(y_test_noisy, model6_predictions)
    #joblib.dump(model6, 'model6.pkl')
    # print(model6_classification_report)
    # print(model6_confusion_matrix)

    print("Testing Decision tree model with noisy data")
    #X_train_noisy, y_train_noisy, X_test_noisy, y_test_noisy = noisyData(files)
    model7 = Decision_Tree(X_train_noisy, y_train_noisy)
    model7scores = crossval(model7, X_train_noisy, y_train_noisy)
    print(model7scores)
    print("average: ", model7scores.mean())

    model7.fit(X_train_noisy, y_train_noisy)
    model7_predictions = model7.predict(X_test_noisy)
    model7_classification_report = classification_report(y_test_noisy, model7_predictions, output_dict=True)
    model7_confusion_matrix = confusion_matrix(y_test_noisy, model7_predictions)
    #joblib.dump(model6, 'model6.pkl')
    # print(model7_classification_report)
    # print(model7_confusion_matrix)

    print("Testing Logistic Regression model with noisy data")
    model8 = Logistic(X_train_noisy, y_train_noisy)
    model8scores = crossval(model8, X_train_noisy, y_train_noisy)
    print(model8scores)
    print("average: ", model8scores.mean())
    model8.fit(X_train_noisy,y_train_noisy)
    model8_predictions = model8.predict(X_test_noisy)
    model8_classification_report = classification_report(y_test_noisy, model8_predictions, output_dict=True)
    model8_confusion_matrix = confusion_matrix(y_test_noisy, model8_predictions)
    #joblib.dump(model6, 'model6.pkl')
    # print(model8_classification_report)
    # print(model8_confusion_matrix)


    print("Testing KNN model with noisy data")
    # X_train_noisy, y_train_noisy, X_test_noisy, y_test_noisy = noisyData(files)
    model9 = Neighbor(X_train_noisy, y_train_noisy)
    model9scores = crossval(model9, X_train_noisy, y_train_noisy)
    print(model9scores)
    print("average: ", model9scores.mean())

    model9.fit(X_train_noisy, y_train_noisy)
    model9_predictions = model9.predict(X_test_noisy)
    model9_classification_report = classification_report(y_test_noisy, model9_predictions, output_dict=True)
    model9_confusion_matrix = confusion_matrix(y_test_noisy, model9_predictions)
    #joblib.dump(model6, 'model6.pkl')
    # print(model9_classification_report)
    # print(model9_confusion_matrix)

    data = {
        'Model': ['classical Noisy', 'classical Clean', 'balanced Noisy', 'balanced Clean', 'Random Forest', 'Naive Bayes', 'Decision tree','Logistic Regression', 'KNN'],
        'Accuracy': [model1scores.mean(), model2scores.mean(), model3scores.mean(), model4scores.mean(), model5scores.mean(), model6scores.mean(), model7scores.mean(), model8scores.mean(), model9scores.mean()]
    }
    df = pd.DataFrame(data)
    plt.bar(data=df, x=data["Model"], height=data["Accuracy"])
    plt.show()

    x = np.arange(9)
    labels = ['classical Noisy', 'classical Clean', 'balanced Noisy', 'balanced Clean', 'Random Forest', 'Naive Bayes', 'Decision tree','Logistic Regression', 'KNN']
    y1 = [model1_classification_report["macro avg"]["precision"], model2_classification_report["macro avg"]["precision"], model3_classification_report["macro avg"]["precision"], model4_classification_report["macro avg"]["precision"], model5_classification_report["macro avg"]["precision"],
    model6_classification_report["macro avg"]["precision"], model7_classification_report["macro avg"]["precision"], model8_classification_report["macro avg"]["precision"], model9_classification_report["macro avg"]["precision"]]
    y2 = [model1_classification_report["macro avg"]["recall"], model2_classification_report["macro avg"]["recall"], model3_classification_report["macro avg"]["recall"], model4_classification_report["macro avg"]["recall"], model5_classification_report["macro avg"]["recall"],
    model6_classification_report["macro avg"]["recall"], model7_classification_report["macro avg"]["recall"], model8_classification_report["macro avg"]["recall"], model9_classification_report["macro avg"]["recall"]]
    y3 = [model1_classification_report["macro avg"]["f1-score"], model2_classification_report["macro avg"]["f1-score"], model3_classification_report["macro avg"]["f1-score"], model4_classification_report["macro avg"]["f1-score"], model5_classification_report["macro avg"]["f1-score"],
    model6_classification_report["macro avg"]["f1-score"], model7_classification_report["macro avg"]["f1-score"], model8_classification_report["macro avg"]["f1-score"], model9_classification_report["macro avg"]["f1-score"]]
    width = 0.2
    
    # plot data in grouped manner of bar type
    plt.bar(x-0.2, y1, width)
    plt.bar(x, y2, width)
    plt.bar(x+0.2, y3, width)
    plt.xticks(x, labels)
    plt.xlabel("Teams")
    plt.ylabel("Scores")
    plt.legend(["avg precision", "avg recall", "avg f1-score"])
    plt.show()