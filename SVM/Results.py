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

test_percentage = 10

for i in range(0,4):

    test_percentage = test_percentage +10
    print(test_percentage)
    Data = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\session 2 results\\Imposter more\\Horizontal_NumberofsamplesTraining' + str(test_percentage) +'_SittingVsSittingSession2.csv')

    Data.columns=['User',	'Scenario',	'True Negative',	'False Positive',	'False Negative',	'True Positive',	'Accuracy Score']

    group = Data.groupby(['Scenario'])

    Scenario1 = group.get_group(1)
    Scenario2 = group.get_group(2)
    Scenario3 = group.get_group(3)

    Scenario1_Mean_Score =Scenario1['Accuracy Score'].mean()
    Scenario2_Mean_Score =Scenario2['Accuracy Score'].mean()
    Scenario3_Mean_Score =Scenario3['Accuracy Score'].mean()

    Scenario1_Median_Score =Scenario1['Accuracy Score'].median()
    Scenario2_Median_Score =Scenario2['Accuracy Score'].median()
    Scenario3_Median_Score =Scenario3['Accuracy Score'].median()


    #Accuracy_Rates = Data_Accuracy_Scores_EqualGenuineandImposter_TestPercentage75.iloc[:, 6]
    #Mean_75_Horizontal = Accuracy_Rates.mean()
    #Median_75_Horizontal = Accuracy_Rates.median()

    #Median_75_Horizontal = Data_Accuracy_Scores_EqualGenuineandImposter_TestPercentage75[6].median()

    #Mean_75_Horizontal_Data_Accuracy_Scores_EqualGenuineandImposter_TestPercentage75 = Data_Accuracy_Scores_EqualGenuineandImposter_TestPercentage75['Accuracy Score'].mean()

    #Median_75_Horizontal_Data_Accuracy_Scores_EqualGenuineandImposter_TestPercentage75 = Data_Accuracy_Scores_EqualGenuineandImposter_TestPercentage75['Accuracy Score'].median()

    #print('Mean_75_Horizontal:',Mean_75_Horizontal)
    #print('Median_75_Horizontal:',Median_75_Horizontal)
    Type_of_swipe ="Vertical"
    Type_of_kernel = "Session 2 - Linear"
    myData_Header = ["Test Percentage","Mean Score - Scenario 1", "Mean Score - Scenario 2", "Mean Score - Scenario 3", "Median Score - SCenario 1", "Median Score - SCenario 2", "Median Score - SCenario 3"]
    myData_s1 = [Type_of_swipe,Type_of_kernel,test_percentage,Scenario1_Mean_Score,Scenario2_Mean_Score, Scenario3_Mean_Score, Scenario1_Median_Score, Scenario2_Median_Score, Scenario3_Median_Score]
    myFile = open('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\Results.csv','a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows([myData_s1])
    myFile.close()


