import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import csv

plt.figure()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], color='navy')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
Genuine_scores = []
Imposter_scores = []
y_true_labels = []

for user in range(102,150):

    if (user!=128) :

        #Vertical
        print("USERRRRRR:", user)
        Data = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\User' + str(user) + '\\df_Training_s1.csv')
        Data_sitting_session2 = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\User' + str(user) + '\\df_Training_s1.csv')
        Data_s2 = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\User' + str(user) + '\\df_Training_s2_session2.csv')
        Data_s3 = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\User' + str(user) + '\\df_Training_s3_session2.csv')

        ############################################################################################################################
        #Getting the class 0 rows
        class0_samples = Data.loc[Data['Class'] == 0]
        class1_samples = Data.loc[Data['Class'] == 1]

        class0_samples_s2 = Data_s2.loc[Data_s2['Class'] == 0]
        class1_samples_s2 = Data_s2.loc[Data_s2['Class'] == 1]

        class0_samples_s3 = Data_s3.loc[Data_s3['Class'] == 0]
        class1_samples_s3 = Data_s3.loc[Data_s3['Class'] == 1]

        class0_samples_sitting_s2 = Data_sitting_session2.loc[Data_sitting_session2['Class'] == 0]
        class1_samples_sitting_s2 = Data_sitting_session2.loc[Data_sitting_session2['Class'] == 1]

        Number_of_class0_samples = class0_samples.shape[0]
        Number_of_class1_samples = class1_samples.shape[0]
        Number_of_class0_samples_s2 = class0_samples_s2.shape[0]
        Number_of_class1_samples_s2 = class1_samples_s2.shape[0]

        Number_of_class0_samples_s3 = class0_samples_s3.shape[0]
        Number_of_class1_samples_s3 = class1_samples_s3.shape[0]

        Number_of_class0_samples_sitting_s2 = class0_samples_sitting_s2.shape[0]
        Number_of_class1_samples_sitting_s2 = class1_samples_sitting_s2.shape[0]

        #Choose imposter samples from randomly chosen user_id between 101 to 150
        Imposter_samples = class1_samples.sample(n=Number_of_class0_samples)
        random_user = random.randint(102, 150)
        if user == random_user:
            random_user = random.randint(102, 150)

        Imposter_samples['user_number'] == random_user

        print('random_user:',random_user)

        Imposter_samples_specific_user = class1_samples.loc[class1_samples['user_number'] == random_user]
        # Pick equal genuine and imposter samples

        Imposter_samples = Imposter_samples_specific_user.sample(n=50)
        Genuine_samples = class0_samples.sample(n=50)

        Number_of_samples = len(Genuine_samples)
        print('Genuine Sample : ',len(Genuine_samples))
        print('Imposter Samples: ', len(Imposter_samples))

        ''' 
        #Pick equal genuine and imposter samples
        Imposter_samples = class1_samples.sample(n=Number_of_class0_samples)
        Genuine_samples = class0_samples
        Imposter_samples_s2 = class1_samples_s2.sample(n=Number_of_class0_samples_s2)
        Genuine_samples_s2 = class0_samples_s2
        Imposter_samples_s3 = class1_samples_s3.sample(n=Number_of_class0_samples_s3)
        Genuine_samples_s3 = class0_samples_s3
        '''

        ########################Scenario 2#####################################################
        # Choose imposter samples from randomly chosen user_id between 101 to 150
        Imposter_samples_sitting_s2 = class1_samples_sitting_s2.sample(n=Number_of_class0_samples_sitting_s2)
        random_user = random.randint(102, 127)

        if user == random_user :
            random_user = random.randint(102, 127)
        Imposter_samples_sitting_s2['user_number'] == random_user
        print('random_user s2:', random_user)
        Imposter_samples_specific_user_sitting_s2 = class1_samples_sitting_s2.loc[class1_samples_sitting_s2['user_number'] == random_user]

        #Number of Imposter samples are more than number of genuine samples
        if len(Imposter_samples_specific_user_sitting_s2) > Number_of_class0_samples_sitting_s2:
            Imposter_samples_sitting_s2 = Imposter_samples_specific_user_sitting_s2.sample(n=Number_of_class0_samples_sitting_s2)
        # Number of Imposter samples are less than number of genuine samples
        if len(Imposter_samples_specific_user_sitting_s2) < Number_of_class0_samples_sitting_s2:
            Imposter_samples_sitting_s2 = Imposter_samples_specific_user_sitting_s2.sample(n=len(Imposter_samples_specific_user_sitting_s2))

        Genuine_samples_sitting_s2 = class0_samples_sitting_s2.sample(n=Number_of_class0_samples_sitting_s2)
        ########################Scenario 2############################################################


        ''' 
        ########################Scenario 3#############################################################
    
        # Choose imposter samples from randomly chosen user_id between 101 to 150
        Imposter_samples_s3 = class1_samples_s3.sample(n=Number_of_class0_samples_s3)
        random_user = random.randint(102, 150)
        if user == random_user:
            random_user = random.randint(102, 150)
        Imposter_samples_s3['user_number'] == random_user
        print('random_user s2:', random_user)
        Imposter_samples_specific_user_s3 = class1_samples_s3.loc[class1_samples_s3['user_number'] == random_user]
    
        # Pick equal genuine and imposter samples
        print('Length Imposter:', len(Imposter_samples_specific_user_s3))
        print('Length Genuine:', Number_of_class0_samples_s3)
    
        # Number of Imposter samples are more than number of genuine samples
        if len(Imposter_samples_specific_user_s3) > Number_of_class0_samples_s3:
            Imposter_samples_s3 = Imposter_samples_specific_user_s3.sample(n=Number_of_class0_samples_s3)
        # Number of Imposter samples are less than number of genuine samples
        if len(Imposter_samples_specific_user_s3) < Number_of_class0_samples_s3:
            Imposter_samples_s3 = Imposter_samples_specific_user_s3.sample(n=len(Imposter_samples_specific_user_s3))
    
        Genuine_samples_s3 = class0_samples_s3.sample(n=Number_of_class0_samples_s3)
    
        ########################Scenario 3############################################################
        '''



        #Dataset with all three scenarios
        Data = Genuine_samples.append(Imposter_samples, ignore_index=True)

       # Data_s2 = Genuine_samples_s2.append(Imposter_samples_s2, ignore_index=True)
        #Data_s3 = Genuine_samples_s3.append(Imposter_samples_s3, ignore_index=True)
        Data_sitting_s2 = Genuine_samples_sitting_s2.append(Imposter_samples_sitting_s2, ignore_index=True)
        ###############################################################################################################################


        Data = Data.dropna()
        Data_s2 = Data_s2.dropna()
        Data_s3 = Data_s3.dropna()
        Data_sitting_s2 = Data_sitting_s2.dropna()


        Data = Data.drop(columns=['user_number'])
        Data_s2 = Data_s2.drop(columns=['user_number'])
        Data_s3 = Data_s3.drop(columns=['user_number'])
        Data_sitting_s2 = Data_sitting_s2.drop(columns=['user_number'])


        #Feature Set
        X = Data.drop('Class', axis=1)
        X_s2 = Data_s2.drop('Class', axis=1)
        X_s3 = Data_s3.drop('Class', axis=1)
        X_sitting_s2 = Data_sitting_s2.drop('Class', axis=1)

        #Class details
        y = Data['Class']
        y_s2 = Data_s2['Class']
        y_s3 = Data_s3['Class']
        y_sitting_s2 = Data_sitting_s2['Class']

        #Define the scaler
        scaler = StandardScaler()

        #sns.lmplot('Total_Stroke_Time','Peak_Velocity_Value',data=Data,hue='Class', palette='Set1', fit_reg=False,scatter_kws={"s":70})
        #plt.show()
        #Splitting the Dataset

        #Scenario 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

        #Scenario 2
        X_s2_train, X_s2_test, y_s2_train, y_s2_test = train_test_split(X_s2, y_s2, test_size = 0.10)

        #Scenario 3
        X_s3_train, X_s3_test, y_s3_train, y_s3_test = train_test_split(X_s3, y_s3, test_size = 0.10)

        # Scenario 3
        X_sitting_s2_train, X_sitting_s2_test, y_sitting_s2_train, y_sitting_s2_test = train_test_split(X_sitting_s2, y_sitting_s2, test_size=0.10)


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

        #Standardise session 2 data
        X_sitting_s2_train = scaler.fit(X_sitting_s2_train).transform(X_sitting_s2_train)
        X_sitting_s2_test = scaler.fit(X_sitting_s2_test).transform(X_sitting_s2_test)


        #Model creation
        svclassifier = SVC(kernel='rbf',probability=True)


        #Model Fitting
        svclassifier.fit(X_train, y_train)
        y_pred = svclassifier.predict(X_test)

       #score = svclassifier.score(X_test,y_test)

        ######################## PCA Selection #########################

        ''' 
        # Make an instance of the Model
        pca = PCA(n_components=2)
        # Fit PCA on training set
        pca.fit(X_train)
        # Apply the mapping (transform) to both the training set and the test set.
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    
        ######################## PCA Selection end ####################
        '''




        ################################# ROC Curve ##################################
        # PREDICT PROBABILITY SCORE = 2D ARRAY FOR EACH PREDICTION
        probas_ = svclassifier.fit(X_train, y_train).predict_proba(X_sitting_s2_test)
        print('Length: ',len(X_sitting_s2_test))
        threshold = 0.5
        print(probas_)

        predicted_genuine = (probas_[:, 1] >= threshold).astype('int')
        predicted_imposter = (probas_[:, 0] >= threshold).astype('int')
        print('predicted_imposter',predicted_genuine)
        print('predicted_imposter',predicted_imposter)

        Genuine_scores.extend(predicted_genuine)
        Imposter_scores.extend(predicted_imposter)

        #Imposter_scores.extend(probas_[:, 0])

        y_true_labels.extend(y_sitting_s2_test)


        print('Genuine Scores:',Genuine_scores)
        print('Imposter Scores:',Imposter_scores)


        #Session 1 - Scenario 1 against Scenario 2
        #probas_ = svclassifier.fit(X_train, y_train).predict_proba(X_s2_test)

        # Session 1 - Scenario 1 against Scenario 2
       # probas_ = svclassifier.fit(X_train, y_train).predict_proba(X_test)

        # Compute ROC curve and area the curve
       # print("X_test length:",len(X_test))

        fpr, tpr, thresholds = roc_curve(y_sitting_s2_test, probas_[:, 1])




        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
       # print("EER:",eer,"thresh",thresh)
        #print("thresholds:",thresholds)
       # print("False Positive Rates:",fpr)
        #print("True Positive Rates:", tpr)
        #print("Threshold:", thresholds)


        #predictedprobSVC = svclassifier.predict_proba(X_test)

        # GET ROC DATA
        #fpr, tpr, thresholds = roc_curve(y_test, predictedprobSVC[:, 1], pos_label=2)
        roc_auc = auc(fpr, tpr)
        # GRAPH DATA

        plt.title('Linear SVM Classifier ROC Sitting Vs Treadmill')
       # plt.plot(fpr, tpr, color='blue', lw=2, label='SVM ROC area = %0.2f)' % roc_auc)
        #plt.plot(fpr, tpr, label='User' + str(user))
      #  plt.legend(loc="lower right", prop={'size': 4})




        #Model fitting Scenario 2
    #    svclassifier.fit(X_s2_train, y_s2_train)
     #   y_pred_s2 = svclassifier.predict(X_s2_test)
        #Model fitting scenario 3


      #  svclassifier.fit(X_s3_train, y_s3_train)
       # y_pred_s3 = svclassifier.predict(X_s3_test)

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



        #myData_Header = ["User","Scenario", "True Negative", "False Positive", "False Negative", "True Positive", "Accuracy Score"]
        myData_s1 = [user, roc_auc,eer,thresh]
        #myData_s2 = [user,2, s2_tn, s2_fp, s2_fn, s2_tp, accuracy_rate_s2]
        #myData_s3 = [user, 3, s3_tn, s3_fp, s3_fn, s3_tp, accuracy_rate_s3]


        ''' 
        myFile = open('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\session 1 results\\equal samples\\Probability_scores\\SVM\\Vertical_NumberofsamplesTraining50NewTesting20_Session1_SittingVsWAlking.csv', 'a')
        with myFile:
           writer = csv.writer(myFile)
           writer.writerows([myData_s1])
           #writer.writerows([myData_s2])
           #writer.writerows([myData_s3])
    
        myFile.close()
        '''


print(Genuine_scores)

fpr, tpr, thresholds = roc_curve(y_true_labels, Genuine_scores)
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)
roc_auc = auc(fpr, tpr)

print(fpr,tpr,thresholds)

print('eer system:',eer,'thresh:',thresh,'roc_auc',roc_auc)

# GRAPH DATA

plt.title('Linear SVM Classifier ROC Sitting Vs Treadmill')
   # plt.plot(fpr, tpr, color='blue', lw=2, label='SVM ROC area = %0.2f)' % roc_auc)
plt.plot(fpr, tpr, label='User' + str(user))
plt.legend(loc="lower right", prop={'size': 4})
plt.show()
#plt.savefig('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\session 1 results\\equal samples\\ROC Curves\\Scenario 1\\Vertical\\ROC_Horizontal_NumberofsamplesTraining15_WalkingSession1VsWalkingSession2.jpg',dpi = 100)