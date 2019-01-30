import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import csv


for user in range(150,151):

    ''' 
    #Horizontal
    print("USERRRRRR:", user)
    Data = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\User' + str(user) + '\\df_Training_s1.csv')
    Data_s2 = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\User' + str(user) + '\\df_Training_s2.csv')
    Data_s3 = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\User' + str(user) + '\\df_Training_s3.csv')
    '''
    #Vertical
    print("USERRRRRR:", user)
    Data = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\User' + str(user) + '\\df_Vertical_Training_s1.csv')
    Data_s2 = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\User' + str(user) + '\\df_Vertical_Training_s2.csv')
    Data_s3 = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\User' + str(user) + '\\df_Vertical_Training_s3.csv')

    ############################################################################################################################
    #Getting the class 0 rows
    class0_samples = Data.loc[Data['Class'] == 0]
    class1_samples = Data.loc[Data['Class'] == 1]

    class0_samples_s2 = Data_s2.loc[Data_s2['Class'] == 0]
    class1_samples_s2 = Data_s2.loc[Data_s2['Class'] == 1]


    class0_samples_s3 = Data_s3.loc[Data_s3['Class'] == 0]
    class1_samples_s3 = Data_s3.loc[Data_s3['Class'] == 1]


    Number_of_class0_samples = class0_samples.shape[0]
    Number_of_class1_samples = class1_samples.shape[0]
    Number_of_class0_samples_s2 = class0_samples_s2.shape[0]
    Number_of_class1_samples_s2 = class1_samples_s2.shape[0]
    Number_of_class0_samples_s3 = class0_samples_s3.shape[0]
    Number_of_class1_samples_s3 = class1_samples_s3.shape[0]

    #Pick equal genuine and imposter samples
    Imposter_samples = class1_samples.sample(n=Number_of_class0_samples)
    Genuine_samples = class0_samples
    Imposter_samples_s2 = class1_samples_s2.sample(n=Number_of_class0_samples_s2)
    Genuine_samples_s2 = class0_samples_s2
    Imposter_samples_s3 = class1_samples_s3.sample(n=Number_of_class0_samples_s3)
    Genuine_samples_s3 = class0_samples_s3

    Data = Genuine_samples.append(Imposter_samples, ignore_index=True)
    Data_s2 = Genuine_samples_s2.append(Imposter_samples_s2, ignore_index=True)
    Data_s3 = Genuine_samples_s3.append(Imposter_samples_s3, ignore_index=True)

    ###############################################################################################################################


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.95)

    #Scenario 2
    X_s2_train, X_s2_test, y_s2_train, y_s2_test = train_test_split(X_s2, y_s2, test_size = 0.95)

    #Scenario 3
    X_s3_train, X_s3_test, y_s3_train, y_s3_test = train_test_split(X_s3, y_s3, test_size = 0.95)



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
    if user!=135 and user!=137 and user!=144:
        X_s3_train = scaler.fit(X_s3_train).transform(X_s3_train)
        X_s3_test = scaler.fit(X_s3_test).transform(X_s3_test)


    #Model creation
    svclassifier = SVC(kernel='linear')


    #Model Fitting
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    #Model fitting Scenario 2
    svclassifier.fit(X_s2_train, y_s2_train)
    y_pred_s2 = svclassifier.predict(X_s2_test)
    #Model fitting scenario 3

    if user != 135 and user!=137 and user!=144:
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

    if user != 135 and user !=137 and user!=144:
        accuracy_rate_s3 = accuracy_score(y_s3_test, y_pred_s3)
        print('Accuracy Score S2: ', accuracy_score(y_s3_test, y_pred_s3))
        print(confusion_matrix(y_s3_test, y_pred_s3))
        print(classification_report(y_s3_test, y_pred_s3))
        s3_tn, s3_fp, s3_fn, s3_tp = confusion_matrix(y_s3_test, y_pred_s3).ravel()




    myData_Header = ["User","Scenario", "True Negative", "False Positive", "False Negative", "True Positive", "Accuracy Score"]
    myData_s1 = [user,1,tn, fp, fn, tp, accuracy_score_s1]
    myData_s2 = [user,2, s2_tn, s2_fp, s2_fn, s2_tp, accuracy_rate_s2]
    myData_s3 = [user, 3, s3_tn, s3_fp, s3_fn, s3_tp, accuracy_rate_s3]

    myFile = open('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\Accuracy_Scores_EqualGenuineandImposter_TestPercentage95_Vertical.csv', 'a')
    with myFile:
       writer = csv.writer(myFile)
       writer.writerows([myData_s1])
       writer.writerows([myData_s2])
       writer.writerows([myData_s3])

    myFile.close()
