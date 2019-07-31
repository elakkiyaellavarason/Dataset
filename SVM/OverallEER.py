import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.model_selection import StratifiedKFold
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


def calculate_diagnostic_performance(actual_predicted):
    """ Calculate sensitivty and specificty.
    Takes a Numpy array of 1 and zero, two columns: actual and predicted
    Returns a tuple of results:
    1) accuracy: proportion of test results that are correct
    2) sensitivity: proportion of true +ve identified
    3) specificity: proportion of true -ve identified
    4) positive likelihood: increased probability of true +ve if test +ve
    5) negative likelihood: reduced probability of true +ve if test -ve
    6) false positive rate: proportion of false +ves in true -ve patients
    7) false negative rate:  proportion of false -ves in true +ve patients
    8) positive predictive value: chance of true +ve if test +ve
    9) negative predictive value: chance of true -ve if test -ve
    10) Count of test positives

    *false positive rate is the percentage of healthy individuals who
    incorrectly receive a positive test result
    * alse neagtive rate is the percentage of diseased individuals who
    incorrectly receive a negative test result

    """
    actual_predicted = test_results.values
    actual_positives = actual_predicted[:, 0] == 1
    actual_negatives = actual_predicted[:, 0] == 0
    test_positives = actual_predicted[:, 1] == 1
    test_negatives = actual_predicted[:, 1] == 0
    test_correct = actual_predicted[:, 0] == actual_predicted[:, 1]
    accuracy = np.average(test_correct)
    true_positives = actual_positives & test_positives
    true_negatives = actual_negatives & test_negatives
    sensitivity = np.sum(true_positives) / np.sum(actual_positives)
    specificity = np.sum(true_negatives) / np.sum(actual_negatives)
    positive_likelihood = sensitivity / (1 - specificity)
    negative_likelihood = (1 - sensitivity) / specificity
    false_postive_rate = 1 - specificity
    false_negative_rate = 1 - sensitivity
    positive_predictive_value = np.sum(true_positives) / np.sum(test_positives)
    negative_predicitive_value = np.sum(true_negatives) / np.sum(test_negatives)
    positive_rate = np.mean(actual_predicted[:, 1])
    return (accuracy, sensitivity, specificity, positive_likelihood,
            negative_likelihood, false_postive_rate, false_negative_rate,
            positive_predictive_value, negative_predicitive_value,
            positive_rate)


for user in range(102,104):

    if (user!=128) :

        #Vertical
        print("USERRRRRR:", user)
        Data = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\User' + str(user) + '\\df_Training_s1.csv')
        Data_sitting_session2 = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\User' + str(user) + '\\df_Training_s1.csv')


        ############################################################################################################################
        #Getting the class 0 rows
        class0_samples = Data.loc[Data['Class'] == 0]
        class1_samples = Data.loc[Data['Class'] == 1]


        class0_samples_sitting_s2 = Data_sitting_session2.loc[Data_sitting_session2['Class'] == 0]
        class1_samples_sitting_s2 = Data_sitting_session2.loc[Data_sitting_session2['Class'] == 1]

        Number_of_class0_samples = class0_samples.shape[0]
        Number_of_class1_samples = class1_samples.shape[0]

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

        #Dataset with all three scenarios
        Data = Genuine_samples.append(Imposter_samples, ignore_index=True)

       # Data_s2 = Genuine_samples_s2.append(Imposter_samples_s2, ignore_index=True)
        #Data_s3 = Genuine_samples_s3.append(Imposter_samples_s3, ignore_index=True)
        Data_sitting_s2 = Genuine_samples_sitting_s2.append(Imposter_samples_sitting_s2, ignore_index=True)
        ###############################################################################################################################

        Data = Data.dropna()
        Data_sitting_s2 = Data_sitting_s2.dropna()
        Data = Data.drop(columns=['user_number'])
        Data_sitting_s2 = Data_sitting_s2.drop(columns=['user_number'])


        #Feature Set
        X = Data.drop('Class', axis=1)
        X_sitting_s2 = Data_sitting_s2.drop('Class', axis=1)

        #Class details
        y = Data['Class']
        y_sitting_s2 = Data_sitting_s2['Class']


        ################################################ K-fold Cross validation #########################################

        # Set up stratified k-fold
        splits = 10
        svm_results_linear = np.zeros((splits, 10))
        skf = StratifiedKFold(n_splits=splits)
        print('skf.get_n_splits(X, y):',skf.get_n_splits(X, y))
        loop_count =0

        X = X.values
        y = y.values

        print('Index values: ',len(X))

        for train_index, test_index in skf.split(X, y):
            print('Split', loop_count + 1, 'out of', splits)
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            sc = StandardScaler()  # new Standard Scalar object
            sc.fit(X_train)
            X_train_std = sc.transform(X_train)
            X_test_std = sc.transform(X_test)
            combined_results = pd.DataFrame()


            # %% SVM (Support Vector Machine) Linear
            svm = SVC(kernel='linear', C=1.0, probability=True)

            ''' 
            probas_ = svm.fit(X_train_std, y_train).predict_proba(X_test_std)
            Genuine_scores.extend(probas_[:, 1])
            Imposter_scores.extend(probas_[:, 0])
            y_true_labels.extend(y_test)
            print('y_sitting_s2_test:', y_test, 'Probability:', probas_[:, 1])
            '''
            svm.fit(X_train_std, y_train)
            y_pred = svm.predict(X_test_std)
            test_results = pd.DataFrame(np.vstack((y_test, y_pred)).T)
            diagnostic_performance = (calculate_diagnostic_performance
                                      (test_results.values))
            svm_results_linear[loop_count, :] = diagnostic_performance
            combined_results['SVM_linear'] = y_pred


            ''' 
            # %% SVM (Support Vector Machine) RBF
            svm = SVC(kernel='rbf', C=1.0)
            svm.fit(X_train_std, y_train)
            y_pred = svm.predict(X_test_std)
            test_results = pd.DataFrame(np.vstack((y_test, y_pred)).T)
            diagnostic_performance = (calculate_diagnostic_performance
                                      (test_results.values))
            svm_results_rbf[loop_count, :] = diagnostic_performance
            combined_results['SVM_rbf'] = y_pred
            '''


            # Increment loop count
            loop_count += 1

        # %% Transfer results to Pandas arrays
        results_summary = pd.DataFrame()

        results_column_names = (['accuracy', 'sensitivity',
                                 'specificity',
                                 'positive likelihood',
                                 'negative likelihood',
                                 'false positive rate',
                                 'false negative rate',
                                 'positive predictive value',
                                 'negative predictive value',
                                 'positive rate'])

        svm_results_lin_df = pd.DataFrame(svm_results_linear)
        svm_results_lin_df.columns = results_column_names
        results_summary['SVM_lin'] = svm_results_lin_df.mean()

        # %% Print summary results
        print('Results Summary:')
        print(results_summary)


fpr, tpr, thresholds = roc_curve(y_true_labels, Genuine_scores)
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)
roc_auc = auc(fpr, tpr)

print('eer system:',eer,'thresh:',thresh,'roc_auc',roc_auc)

# GRAPH DATA

plt.title('Linear SVM Classifier ROC Sitting Vs Treadmill')
   # plt.plot(fpr, tpr, color='blue', lw=2, label='SVM ROC area = %0.2f)' % roc_auc)
plt.plot(fpr, tpr, label='User' + str(user))
plt.legend(loc="lower right", prop={'size': 4})
plt.show()
#plt.savefig('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\session 1 results\\equal samples\\ROC Curves\\Scenario 1\\Vertical\\ROC_Horizontal_NumberofsamplesTraining15_WalkingSession1VsWalkingSession2.jpg',dpi = 100)