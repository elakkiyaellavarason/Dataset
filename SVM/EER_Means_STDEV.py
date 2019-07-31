import pandas as pd

test_percentage = [15,24,30,40]

for i in range(0,4):



    Data = pd.read_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\session 1 results\\equal samples\\Probability_scores\\Naive Bayes\\session 2\\Vertical_NumberofsamplesTraining' + str(test_percentage[i]) + '_SittingVsBus.csv', header=None)

    Data.columns = ['1', '2' , '3','4' ]

    MEdian = Data['3'].median()
    MEan = Data['3'].mean()
    std = Data['3'].std()
    sem = Data['3'].sem()


    print(str(round(MEan, 2))  )


