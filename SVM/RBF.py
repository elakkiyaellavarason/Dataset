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


EER_list_1 =[]
EER_list_2= []
EER_list_3 = []
EER_list_4 = []
EER_list_5 = []

index = 0

for user in range(138, 150):

    index = index +1

    print("USERRRRRR:", user)
    Data = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\User' + str( user) + '\\df_Training_s1.csv')
    Data_sitting_session2 = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\User' + str(user) + '\\df_Training_s3.csv')

    ############################################################################################################################
    # Getting the class 0 rows
    class0_samples = Data.loc[Data['Class'] == 0]
    class1_samples = Data.loc[Data['Class'] == 1]
    class0_samples_sitting_s2 = Data_sitting_session2.loc[Data_sitting_session2['Class'] == 0]
    class1_samples_sitting_s2 = Data_sitting_session2.loc[Data_sitting_session2['Class'] == 1]

    ########################Scenario 1#####################################################
    Number_of_class0_samples = class0_samples.shape[0]
    Number_of_class1_samples = class1_samples.shape[0]

    Number_of_class0_samples_sitting_s2 = class0_samples_sitting_s2.shape[0]
    Number_of_class1_samples_sitting_s2 = class1_samples_sitting_s2.shape[0]

    # Choose imposter samples from randomly chosen user_id between 101 to 150
    Imposter_samples = class1_samples.sample(n=Number_of_class0_samples)
    random_user = random.randint(102, 150)
    if user == random_user:
        random_user = random.randint(102, 150)

    Imposter_samples['user_number'] == random_user

    print('random_user:', random_user)

    Imposter_samples_specific_user = class1_samples.loc[class1_samples['user_number'] == random_user]
    # Pick equal genuine and imposter samples

    Imposter_samples = Imposter_samples_specific_user.sample(n=40)
    Genuine_samples = class0_samples.sample(n=40)

    Number_of_samples = len(Genuine_samples)
    print('Genuine Sample : ', len(Genuine_samples))
    print('Imposter Samples: ', len(Imposter_samples))
    ########################Scenario 1 End#####################################################

    ########################Scenario 2#####################################################
    # Choose imposter samples from randomly chosen user_id between 101 to 150
    Imposter_samples_sitting_s2 = class1_samples_sitting_s2.sample(n=Number_of_class0_samples_sitting_s2)
    random_user = random.randint(102, 149)

    if user == random_user:
        random_user = random.randint(102, 149)
    Imposter_samples_sitting_s2['user_number'] == random_user
    print('random_user s2:', random_user)
    Imposter_samples_specific_user_sitting_s2 = class1_samples_sitting_s2.loc[class1_samples_sitting_s2['user_number'] == random_user]

    # Number of Imposter samples are more than number of genuine samples
    if len(Imposter_samples_specific_user_sitting_s2) > Number_of_class0_samples_sitting_s2:
        Imposter_samples_sitting_s2 = Imposter_samples_specific_user_sitting_s2.sample(
            n=Number_of_class0_samples_sitting_s2)
    # Number of Imposter samples are less than number of genuine samples
    if len(Imposter_samples_specific_user_sitting_s2) < Number_of_class0_samples_sitting_s2:
        Imposter_samples_sitting_s2 = Imposter_samples_specific_user_sitting_s2.sample(
            n=len(Imposter_samples_specific_user_sitting_s2))

    Genuine_samples_sitting_s2 = class0_samples_sitting_s2.sample(n=Number_of_class0_samples_sitting_s2)
    ########################Scenario 2############################################################


   #######################################################################################################################################
    # Dataset with all three scenarios
    Data = Genuine_samples.append(Imposter_samples, ignore_index=True)
    Data_sitting_s2 = Genuine_samples_sitting_s2.append(Imposter_samples_sitting_s2, ignore_index=True)
    ###############################################################################################################################

    Data = Data.dropna()
    Data_sitting_s2 = Data_sitting_s2.dropna()

    Data = Data.drop(columns=['user_number'])
    Data_sitting_s2 = Data_sitting_s2.drop(columns=['user_number'])

    # Feature Set
    X = Data.drop('Class', axis=1)
    X_sitting_s2 = Data_sitting_s2.drop('Class', axis=1)

    # Class details
    y = Data['Class']
    y_sitting_s2 = Data_sitting_s2['Class']

    # Define the scaler
    scaler = StandardScaler()

    # Splitting the Dataset
    # Scenario 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75)

    # Scenario 3
    X_sitting_s2_train, X_sitting_s2_test, y_sitting_s2_train, y_sitting_s2_test = train_test_split(X_sitting_s2,y_sitting_s2, test_size=0.75)



    # Standardise the data
    X_train = scaler.fit(X_train).transform(X_train)
    X_test = scaler.fit(X_test).transform(X_test)

    # Standardise session 2 data
    X_sitting_s2_train = scaler.fit(X_sitting_s2_train).transform(X_sitting_s2_train)
    X_sitting_s2_test = scaler.fit(X_sitting_s2_test).transform(X_sitting_s2_test)

    # Model creation

    #gammas = [0.1, 1, 10, 100]
    cs = [0.1, 1, 10, 100, 1000]
    for c in cs:
        svclassifier = SVC(kernel='rbf', C = c, probability=True)
        svclassifier.fit(X_train, y_train)


        ################################# ROC Curve ##################################
        # PREDICT PROBABILITY SCORE = 2D ARRAY FOR EACH PREDICTION
        probas_ = svclassifier.fit(X_train, y_train).predict_proba(X_sitting_s2_test)

        # Compute ROC curve and area the curve
        print(probas_[:, 1])
        fpr, tpr, thresholds = roc_curve(y_sitting_s2_test, probas_[:, 1])

        #eer = thresholds(np.argmin[abs[tpr - fpr]])

        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
        roc_auc = auc(fpr, tpr)

        index_cs = cs.index(c)
        print(index_cs)

        if index_cs==0:
            EER_list_1.insert(index,eer)
        elif index_cs==1:
            EER_list_2.insert(index,eer)
        elif index_cs==2:
            EER_list_3.insert(index, eer)
        elif index_cs==3:
            EER_list_4.insert(index, eer)
        elif index_cs == 4:
            EER_list_5.insert(index, eer)

        plt.title('RBF SVM Classifier ROC Curve for C Parameter ' + str(c))
        # plt.plot(fpr, tpr, color='blue', lw=2, label='SVM ROC area = %0.2f)' % roc_auc)
        plt.plot(fpr, tpr, label='User' + str(user))
        plt.legend(loc="lower right", prop={'size': 4})
       # plt.show()

#EER_list_1.to_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\session 1 results\\equal samples\\Probability_scores\\SVM\\Horizontal_NumberofsamplesTraining15_SittingVsSitting_RBF_C0.1.csv')

print(EER_list_1)
print(EER_list_2)
print(EER_list_3)
print(EER_list_4)
print(EER_list_5)
   # myData_s1 = [user, roc_auc,eer,thresh]
myFile = open('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\session 1 results\\equal samples\\Probability_scores\\SVM\\Horizontal_NumberofsamplesTraining40_SittingVsWalking_RBF_EERcalculationdifferent_2.csv', 'a')
with myFile:
    writer = csv.writer(myFile)
    for val in zip(EER_list_1, EER_list_2, EER_list_3, EER_list_4, EER_list_5):
        writer.writerow(val)

myFile.close()
#plt.show()
# plt.savefig('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\session 1 results\\equal samples\\ROC Curves\\Scenario 1\\Vertical\\ROC_Horizontal_NumberofsamplesTraining15_WalkingSession1VsWalkingSession2.jpg',dpi = 100)