import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import csv



for user in range(102,107):

    Data = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\User' + str(user) + '\\df_Training_s1_session2.csv')
    Data_s2 = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\User' + str(user) + '\\df_Training_s2_session2.csv')
    Data_s3 = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\User' + str(user) + '\\df_Training_s3_session2.csv')

    Data = Data.dropna()
    Data_s2 = Data_s2.dropna()
    Data_s3 = Data_s3.dropna()

    Data = Data.drop(columns=['user_number'])
    Data_s2 = Data_s2.drop(columns=['user_number'])
    Data_s3 = Data_s3.drop(columns=['user_number'])

    #Feature Set
    X = Data.drop('Class', axis=1)
    X_s2 = Data_s2.drop('Class', axis=1)
    X_s3 = Data_s3.drop('Class', axis=1)

    #Class details
    y = Data['Class']
    y_s2 = Data_s2['Class']
    y_s3 = Data_s3['Class']

    #Define the scaler
    scaler = StandardScaler()

    #sns.lmplot('Total_Stroke_Time','Peak_Velocity_Value',data=Data,hue='Class', palette='Set1', fit_reg=False,scatter_kws={"s":70})
    #plt.show()
    #Splitting the Dataset

    #Scenario 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.85)

    #Scenario 2
    X_s2_train, X_s2_test, y_s2_train, y_s2_test = train_test_split(X_s2, y_s2, test_size = 0.85)

    #Scenario 3
    X_s3_train, X_s3_test, y_s3_train, y_s3_test = train_test_split(X_s3, y_s3, test_size = 0.85)



    ''' 
    X_train = X_train.as_matrix().astype(np.float)
    X_test = X_test.as_matrix().astype(np.float)
    y_test = y_test.as_matrix().astype(np.float)
    y_train = y_train.as_matrix().astype(np.float)
    '''

    #Standardise the data
    X_train = scaler.fit(X_train).transform(X_train)
    X_test = scaler.fit(X_test).transform(X_test)

    #Standardise scenario 2
    X_s2_train = scaler.fit(X_s2_train).transform(X_s2_train)
    X_s2_test = scaler.fit(X_s2_test).transform(X_s2_test)

    #Standardise scenario 3
    X_s3_train = scaler.fit(X_s3_train).transform(X_s3_train)
    X_s3_test = scaler.fit(X_s3_test).transform(X_s3_test)


    #Model creation
    svclassifier = SVC(kernel='linear',probability=True)


    #Model Fitting
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    ################################# ROC Curve ##################################

    # PREDICT PROBABILITY SCORE = 2D ARRAY FOR EACH PREDICTION
    predictedprobSVC = svclassifier.predict_proba(X_test)
    # GET ROC DATA
    fpr, tpr, thresholds = roc_curve(y_pred, predictedprobSVC[:, 1], pos_label=2)
    roc_auc = auc(fpr, tpr)
    # GRAPH DATA
    plt.figure()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('SVM Classifier ROC')
    plt.plot(fpr, tpr, color='blue', lw=2, label='SVM ROC area = %0.2f)' % roc_auc)
    plt.legend(loc="lower right")
    plt.show()


    ''' 
    #Score function for ROC curve
    y_score = svclassifier.fit(X_train, y_train).decision_function(X_test)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    ################################# ROC Curve ##################################

    '''









    #Model fitting Scenario 2
    svclassifier.fit(X_s2_train, y_s2_train)
    y_pred_s2 = svclassifier.predict(X_s2_test)
    #Model fitting scenario 3


    svclassifier.fit(X_s3_train, y_s3_train)
    y_pred_s3 = svclassifier.predict(X_s3_test)

    ''' 
    # Get the separating hyperplane
    w = svclassifier.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(30, 60)
    yy = a * xx - (svclassifier.intercept_[0]) / w[1]
    
    # Plot the parallels to the separating hyperplane that pass through the support vectors
    b = svclassifier.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = svclassifier.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])
    
    sns.lmplot('Width',	'Height',data=Data,hue='Class', palette='Set1', fit_reg=False,scatter_kws={"s":70})
    plt.plot(xx, yy, linewidth=2, color='black');
    plt.show()
    '''

    #Scenario 1
    accuracy_score_s1 = (accuracy_score(y_test,y_pred))
    print('Accuracy Score: ',accuracy_score_s1)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    print('true positive:',tp,tn,fp,fn)

    #Scenario 2
    accuracy_rate_s2 = accuracy_score(y_s2_test, y_pred_s2)
    print('Accuracy Score S2: ',accuracy_score(y_s2_test,y_pred_s2))
    print(confusion_matrix(y_s2_test,y_pred_s2))
    print(classification_report(y_s2_test,y_pred_s2))
    s2_tn, s2_fp, s2_fn, s2_tp = confusion_matrix(y_s2_test,y_pred_s2).ravel()


    #Scenario 3

    accuracy_rate_s3 = accuracy_score(y_s3_test, y_pred_s3)
    print('Accuracy Score S2: ', accuracy_score(y_s3_test, y_pred_s3))
    print(confusion_matrix(y_s3_test, y_pred_s3))
    print(classification_report(y_s3_test, y_pred_s3))
    s3_tn, s3_fp, s3_fn, s3_tp = confusion_matrix(y_s3_test, y_pred_s3).ravel()

    print("USERRRRRR:",user)

    myData_Header = ["User","Scenario", "True Negative", "False Positive", "False Negative", "True Positive", "Accuracy Score"]
    myData_s1 = [user,1,tn, fp, fn, tp, accuracy_score_s1]
    myData_s2 = [user,2, s2_tn, s2_fp, s2_fn, s2_tp, accuracy_rate_s2]
    myData_s3 = [user, 3, s3_tn, s3_fp, s3_fn, s3_tp, accuracy_rate_s3]

    myFile = open('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\Accuracy_Scores_TestPercentage90_Horizontal_Linear_Session2.csv', 'a')
    with myFile:
       writer = csv.writer(myFile)
       writer.writerows([myData_s1])
       writer.writerows([myData_s2])
       writer.writerows([myData_s3])

    myFile.close()