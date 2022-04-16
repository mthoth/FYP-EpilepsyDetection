from my_constants import PREPROCESSED_DIR
from prepareData import noisyData, cleanData

import os
import joblib
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut


def SVM_classical(X, y):
    model = svm.SVC(C=1, kernel='rbf')
    # model.fit(X, y)
    return model


def SVM_CLASSWEIGHT_BALANCED(X, y):
    model = svm.SVC(C=1, class_weight='balanced', kernel='rbf')
    # model.fit(X, y)
    return model


def crossval(model, X, y):
    scores = cross_val_score(model, X, y, cv=10, scoring="recall")
    return scores


def best_model(models):
    # TODO: implementation
    return 0


if __name__ == "__main__":
    files = [f for f in os.listdir(PREPROCESSED_DIR) if '.csv' in f]
    print("Testing classical model with noisy data")
    X_train_noisy, y_train_noisy, X_test_noisy, y_test_noisy = noisyData(files)
    model1 = SVM_classical(X_train_noisy, y_train_noisy)
    model1scores = crossval(model1, X_test_noisy, y_test_noisy)
    model1.fit(X_train_noisy, y_train_noisy)
    print(model1scores)
    print("average: ", model1scores.mean())

    model1_predictions = model1.predict(X_test_noisy)
    model1_classification_report1 = classification_report(
        y_test_noisy, model1_predictions)
    model1_confusion_matrix = confusion_matrix(
        y_test_noisy, model1_predictions)
    print(model1_classification_report1)
    print(model1_confusion_matrix)

    print("Testing classical model with clean data")
    X_train_clean, y_train_clean, X_test_clean, y_test_clean = cleanData(files)
    model2 = SVM_classical(X_train_clean, y_train_clean)
    model2scores = crossval(model2, X_test_clean, y_test_clean)
    model2.fit(X_train_clean, y_train_clean)
    print(model2scores)
    print("average: ", model2scores.mean())

    model2_predictions = model2.predict(X_test_clean)
    model2_classification_report = classification_report(
        y_test_clean, model2_predictions)
    model2_confusion_matrix = confusion_matrix(
        y_test_clean, model2_predictions)
    print(model2_classification_report)
    print(model2_confusion_matrix)

    print("Testing balanced model with noisy data")
    # X_train_noisy, y_train_noisy, X_test_noisy, y_test_noisy = noisyData(files)
    model3 = SVM_CLASSWEIGHT_BALANCED(X_train_noisy, y_train_noisy)
    model3scores = crossval(model3, X_test_noisy, y_test_noisy)
    model3.fit(X_train_noisy, y_train_noisy)
    print(model3scores)
    print("average: ", model3scores.mean())

    model3_predictions = model3.predict(X_test_noisy)
    model3_classification_report = classification_report(
        y_test_noisy, model3_predictions)
    model3_confusion_matrix = confusion_matrix(
        y_test_noisy, model3_predictions)
    joblib.dump(model3, 'senior13.pkl')
    print(model3_classification_report)
    print(model3_confusion_matrix)

    print("Testing balanced model with clean data")
    # X_train_clean, y_train_clean, X_test_clean, y_test_clean = cleanData(files)
    model4 = SVM_CLASSWEIGHT_BALANCED(X_train_clean, y_train_clean)
    model4scores = crossval(model4, X_test_clean, y_test_clean)
    model4.fit(X_train_clean, y_train_clean)
    print(model4scores)
    print("average: ", model4scores.mean())
    model4_predictions = model4.predict(X_test_clean)
    model4_classification_report = classification_report(
        y_test_clean, model4_predictions)
    model4_confusion_matrix = confusion_matrix(
        y_test_clean, model4_predictions)
    print(model4_classification_report)
    print(model4_confusion_matrix)
