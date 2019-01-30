import pandas as pd

test_percentage = 0

for i in range(0,5):

    test_percentage = test_percentage + 10

    Data = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\Evaluation\\Session 1\\session 1 results\\equal samples\\Probability_scores\\Horizontal_NumberofsamplesTraining' + str(test_percentage) +'_WalkingSession1VsWalkingSession2.csv', header=None)

    Data.columns = ['User', 'Scenario', 'AUC' , 'EER','threshold' ]

    Mean_Score =Data['EER'].mean()
    Standard_Dev = Data['EER'].std()
    Standard_error = Data['EER'].sem()

    print( str(Standard_error))
    #print(str(test_percentage) + ":" + str(Standard_Dev))





