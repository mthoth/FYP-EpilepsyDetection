import os

import joblib
from sklearn import svm
from prepareData import noisyData, cleanData
from sklearn.metrics import classification_report, confusion_matrix

def SVM_classical(X, y):
    model = svm.SVC(C=1, kernel='rbf')
    model.fit(X, y)
    return model

def SVM_CLASSWEIGHT_BALANCED(X, y):
    model = svm.SVC(C=1, class_weight='balanced', kernel='rbf')
    model.fit(X, y)
    return model

if __name__ == "__main__":
    
    files = [f for f in os.listdir(r'C:\Users\Dell\PycharmProjects\pythonProject\chb10preprocessed') if '.csv' in f]
    print("Testing classical model with noisy data")
    X_train_noisy, y_train_noisy, X_test_noisy, y_test_noisy = noisyData(files)
    model1 = SVM_classical(X_train_noisy, y_train_noisy)
    model1_predictions = model1.predict(X_test_noisy)
    model1_classification_report1 = classification_report(y_test_noisy, model1_predictions)
    model1_confusion_matrix = confusion_matrix(y_test_noisy, model1_predictions)
    print(model1_classification_report1)
    print(model1_confusion_matrix)

    print("Testing classical model with clean data")
    X_train_clean, y_train_clean, X_test_clean, y_test_clean = cleanData(files)
    model2 = SVM_classical(X_train_clean, y_train_clean)
    model2_predictions = model2.predict(X_test_clean)
    model2_classification_report = classification_report(y_test_clean, model2_predictions)
    model2_confusion_matrix = confusion_matrix(y_test_noisy, model2_predictions)
    print(model2_classification_report)
    print(model2_confusion_matrix)

    print("Testing balanced model with noisy data")
    # X_train_noisy, y_train_noisy, X_test_noisy, y_test_noisy = noisyData(files)
    model3 = SVM_CLASSWEIGHT_BALANCED(X_train_noisy, y_train_noisy)
    model3_predictions = model3.predict(X_test_noisy)
    model3_classification_report = classification_report(y_test_noisy, model3_predictions)
    model3_confusion_matrix = confusion_matrix(y_test_noisy, model3_predictions)
    print(model3_classification_report)
    print(model3_confusion_matrix)

    print("Testing balanced model with clean data")
    # X_train_clean, y_train_clean, X_test_clean, y_test_clean = cleanData(files)
    model4 = SVM_CLASSWEIGHT_BALANCED(X_train_clean, y_train_clean)
    with open("SVM-model.pickle", "wb") as f:
        joblib.dump(model4, 'Seniorproject.pkl')
    model4_predictions = model4.predict(X_test_clean)
    model4_classification_report = classification_report(y_test_clean, model4_predictions)
    model4_confusion_matrix = confusion_matrix(y_test_clean, model4_predictions)
    print(model4_classification_report)
    print(model4_confusion_matrix)
