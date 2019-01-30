import numpy as np
from cv2 import norm
import pandas as pd
from numpy import shape
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from dtw import dtw
from pandas import Series
from sklearn.preprocessing import MinMaxScaler, StandardScaler



#slicing rows
#for 1 user



for user in range(101,150):

    df_TrainingAllUsers_H_S1_a = pd.DataFrame()
    df_TrainingAllUsers_H_S2_b = pd.DataFrame()
    df_TrainingAllUsers_H_S3_c = pd.DataFrame()
    df_TrainingAllUsers_H_S1_aa = pd.DataFrame()
    df_TrainingAllUsers_H_S2_bb = pd.DataFrame()
    df_TrainingAllUsers_H_S3_cc = pd.DataFrame()
    df_Training_User141_s1 = pd.DataFrame()
    df_Training_User141_s2 =pd.DataFrame()
    df_Training_User141_s3 = pd.DataFrame()

    for super_user in range(101,150):

        df = pd.read_csv('C:\\Users\\ee244\\Desktop\\Data Collection\\Data\\user' + str(super_user) + '\\session 2\\features\\Vertical_Features_S1.csv', parse_dates=[2])
        df.dropna()

        ''' 
         # Normalised X and Y
        X = df[' X ']
        Y = df[' Y ']
        Delta_X = df['Delta_X']
        Delta_Y = df['Delta_Y']
        
        df['Delta_X'] = df['Delta_X'].fillna(0)
        df['Delta_Y'] = df['Delta_Y'].fillna(0)
        
        # Normalize time series data
        values = X.values.reshape((len(X), 1))
        sample2 = Y.values.reshape((len(Y), 1))
        DeltaX_values = Delta_X.values.reshape(len(Delta_X),1)
        DeltaY_values = Delta_Y.values.reshape(len(Delta_Y), 1)
        
        # train the normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler1 = scaler.fit(values)
        scaler2 = scaler.fit(sample2)
        
        # normalize the dataset and print the first 5 rows
        normalized_X = scaler.transform(values)
        normalized_Y = scaler.transform(sample2)
        normalized_DeltaX = scaler.transform(DeltaX_values)
        normalized_DeltaY = scaler.transform(DeltaY_values)
        
        df['Norm_X'] = normalized_X
        df['Norm_Y'] = normalized_Y
        df['User'] = user
        df['Norm_DeltaX'] = normalized_DeltaX
        df['Norm_DeltaY'] = normalized_DeltaY
        
        '''

        df['user_number'] = super_user
        if super_user == user:
            df['Class'] = 0
        else:
            df['Class'] = 1

        # Separating Horizontal stroke based on scenarios
        '''
        df_h_s1_s1 = df[(df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_1 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_2 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_3 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_4 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_5 ') | ( df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_6 ')]
    
        df_h_s1_s2 = df[
                (df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_1 ') | (df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_2 ') | (
                            df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_3 ') | (
                            df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_4 ') | (
                            df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_5 ') | (
                            df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_6 ')]
        df_h_s1_s3 = df[(df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_1 ') | (df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_2 ') | (
                            df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_3 ') | (
                            df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_4 ') | (
                            df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_5 ') | (
                            df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_6 ')]
        '''
        '''
        #Session 2
        df_h_s1_s1 = df[(df['Page'] == 'S2_S1_Horizontal_Swipe_Activity_1 ') | (df['Page'] == 'S2_S1_Horizontal_Swipe_Activity_2 ') | (df['Page'] == 'S2_S1_Horizontal_Swipe_Activity_3 ') | (df['Page'] == 'S2_S1_Horizontal_Swipe_Activity_4 ') | (df['Page'] == 'S2_S1_Horizontal_Swipe_Activity_5 ') | (df['Page'] == 'S2_S1_Horizontal_Swipe_Activity_6 ') ]

        df_h_s1_s2 = df[(df['Page'] == 'S2_S2_Horizontal_Swipe_Activity_1 ') | (df['Page'] == 'S2_S2_Horizontal_Swipe_Activity_2 ') | (df['Page'] == 'S2_S2_Horizontal_Swipe_Activity_3 ') | (df['Page'] == 'S2_S2_Horizontal_Swipe_Activity_4 ') | (df['Page'] == 'S2_S2_Horizontal_Swipe_Activity_5 ') ]

        df_h_s1_s3 = df[ (df['Page'] == 'Session2_Scenario_3_Horizontal_Swipe_Activity_1 ') | (df['Page'] == 'Session2_Scenario_3_Horizontal_Swipe_Activity_2 ') | (df['Page'] == 'Session2_Scenario_3_Horizontal_Swipe_Activity_3 ') | ( df['Page'] == 'Session2_Scenario_3_Horizontal_Swipe_Activity_4 ')]

        '''



        #Vertical Swipes
        df_h_s1_s1 = df[(df['Page'] == 'S2_S1_Vertical_Scroll_1 ') | (df['Page'] == 'S2_S1_Vertical_Scroll_2 ') | (df['Page'] == 'S2_S1_Vertical_Scroll_3 ') | (df['Page'] == 'S2_S1_Vertical_Scroll_4 ') | (df['Page'] == 'S2_S1_Vertical_Scroll_5 ') | (df['Page'] == 'S2_S1_Vertical_Scroll_6 ') ]

        df_h_s1_s2 = df[(df['Page'] == 'S2_S2_VerticalSwipe_Activity_1 ') | (df['Page'] == 'S1_S2_Vertical_2 ') | (df['Page'] == 'S1_S2_Vertical_3 ') | (df['Page'] == 'S1_S2_Vertical_4 ') ]

        df_h_s1_s3 =   df[ (df['Page'] == 'Scenario_3_VerticalSwipe_Activity_1 ') | (df['Page'] == 'Scenario_3_VerticalSwipe_Activity_2 ') | (df['Page'] == 'Scenario_3_VerticalSwipe_Activity_3 ') | (df['Page'] == 'Scenario_3_VerticalSwipe_Activity_4 ')]

        df_Training_User141_s1 = df_h_s1_s1[['Class','user_number', 'Page', ' Touch_Action ', 'Total_Stroke_Time',	'Peak_Velocity_Value',	'Mid-Location',	'Mean_of_velocity_firsthalf_stroke',	'Mean_of_velocity_secondhalf_stroke',	'Standard-Deviation',	'Average_Acceleration', 'Inking_Distance',	'Width', 'Height',	'Min_Slope',	'Max Slope', 'Start_X',	'End_X',	'Start_Y',	'End_Y','Average_Delta_X',	'Average_Delta_Y',	'Finger_Size_FingerDown',	'Finger_Size_FingerUp',	'Average_Finger_Size',	'Stroke_Area_Outer',	'Average_Velocity','Attack_Angle','Leaving_Angle',	'Number_of_Datapoints']]

        df_Training_User141_s2 = df_h_s1_s2[['Class', 'user_number', 'Page', ' Touch_Action ', 'Total_Stroke_Time', 'Peak_Velocity_Value', 'Mid-Location','Mean_of_velocity_firsthalf_stroke', 'Mean_of_velocity_secondhalf_stroke', 'Standard-Deviation','Average_Acceleration', 'Inking_Distance', 'Width', 'Height', 'Min_Slope', 'Max Slope', 'Start_X', 'End_X','Start_Y', 'End_Y', 'Average_Delta_X', 'Average_Delta_Y', 'Finger_Size_FingerDown', 'Finger_Size_FingerUp',
                 'Average_Finger_Size', 'Stroke_Area_Outer', 'Average_Velocity', 'Attack_Angle', 'Leaving_Angle',
                 'Number_of_Datapoints']]

        df_Training_User141_s3 = df_h_s1_s3[['Class', 'user_number', 'Page', ' Touch_Action ', 'Total_Stroke_Time', 'Peak_Velocity_Value', 'Mid-Location','Mean_of_velocity_firsthalf_stroke', 'Mean_of_velocity_secondhalf_stroke', 'Standard-Deviation','Average_Acceleration', 'Inking_Distance', 'Width', 'Height', 'Min_Slope', 'Max Slope', 'Start_X', 'End_X','Start_Y', 'End_Y', 'Average_Delta_X', 'Average_Delta_Y', 'Finger_Size_FingerDown', 'Finger_Size_FingerUp','Average_Finger_Size', 'Stroke_Area_Outer', 'Average_Velocity', 'Attack_Angle', 'Leaving_Angle', 'Number_of_Datapoints']]


        #Taking only global features
        df_Training_User141_s1 = df_Training_User141_s1[df_Training_User141_s1[' Touch_Action ']== ' ACTION_DOWN ']
        df_Training_User141_s2 = df_Training_User141_s2[df_Training_User141_s2[' Touch_Action '] == ' ACTION_DOWN ']
        df_Training_User141_s3 = df_Training_User141_s3[df_Training_User141_s3[' Touch_Action '] == ' ACTION_DOWN ']

        df_TrainingAllUsers_H_S1 = df_Training_User141_s1[ ['Class','user_number', 'Total_Stroke_Time', 'Peak_Velocity_Value', 'Mid-Location','Mean_of_velocity_firsthalf_stroke', 'Mean_of_velocity_secondhalf_stroke', 'Standard-Deviation','Average_Acceleration', 'Inking_Distance', 'Width', 'Height', 'Min_Slope', 'Max Slope', 'Start_X', 'End_X','Start_Y', 'End_Y', 'Average_Delta_X', 'Average_Delta_Y', 'Finger_Size_FingerDown', 'Finger_Size_FingerUp','Average_Finger_Size', 'Stroke_Area_Outer', 'Average_Velocity', 'Attack_Angle', 'Leaving_Angle','Number_of_Datapoints']]

        df_TrainingAllUsers_H_S2 = df_Training_User141_s2[['Class', 'user_number','Total_Stroke_Time', 'Peak_Velocity_Value', 'Mid-Location','Mean_of_velocity_firsthalf_stroke', 'Mean_of_velocity_secondhalf_stroke', 'Standard-Deviation', 'Average_Acceleration', 'Inking_Distance', 'Width', 'Height', 'Min_Slope', 'Max Slope', 'Start_X', 'End_X','Start_Y', 'End_Y', 'Average_Delta_X', 'Average_Delta_Y', 'Finger_Size_FingerDown', 'Finger_Size_FingerUp','Average_Finger_Size', 'Stroke_Area_Outer', 'Average_Velocity', 'Attack_Angle', 'Leaving_Angle','Number_of_Datapoints']]

        df_TrainingAllUsers_H_S3 = df_Training_User141_s3[['Class','user_number', 'Total_Stroke_Time', 'Peak_Velocity_Value', 'Mid-Location','Mean_of_velocity_firsthalf_stroke', 'Mean_of_velocity_secondhalf_stroke', 'Standard-Deviation','Average_Acceleration', 'Inking_Distance', 'Width', 'Height', 'Min_Slope', 'Max Slope', 'Start_X', 'End_X', 'Start_Y', 'End_Y', 'Average_Delta_X', 'Average_Delta_Y', 'Finger_Size_FingerDown', 'Finger_Size_FingerUp','Average_Finger_Size', 'Stroke_Area_Outer', 'Average_Velocity', 'Attack_Angle', 'Leaving_Angle','Number_of_Datapoints']]

        df_TrainingAllUsers_H_S1_a = df_TrainingAllUsers_H_S1_a.append(df_TrainingAllUsers_H_S1)
        df_TrainingAllUsers_H_S2_b = df_TrainingAllUsers_H_S2_b.append(df_TrainingAllUsers_H_S2)
        df_TrainingAllUsers_H_S3_c = df_TrainingAllUsers_H_S3_c.append(df_TrainingAllUsers_H_S3)

        ''' 
        df_TrainingAllUsers_H_S1_aa = df_TrainingAllUsers_H_S1_aa.append(df_TrainingAllUsers_H_S1_a)
        df_TrainingAllUsers_H_S2_bb = df_TrainingAllUsers_H_S2_bb.append(df_TrainingAllUsers_H_S2_b)
        df_TrainingAllUsers_H_S3_cc = df_TrainingAllUsers_H_S3_cc.append(df_TrainingAllUsers_H_S3_c)
        '''
    df_TrainingAllUsers_H_S1_a.to_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\User' + str(user) + '\\df_session2_s1_Vertical.csv', index=False)
    df_TrainingAllUsers_H_S2_b.to_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\User' + str(user) + '\\df_session2_S2_Vertical.csv', index=False)
    df_TrainingAllUsers_H_S3_c.to_csv('C:\\Users\\ee244\\Desktop\\PhD\\Analysis\\SVM\\Evaluation\\Session 1\\User' + str(user) + '\\df_session2_s3_Vertical.csv', index=False)

    del df_TrainingAllUsers_H_S1_a
    del df_TrainingAllUsers_H_S2_b
    del df_TrainingAllUsers_H_S3_c






# df_Training_H_S3 = df_h_s1_s3[['User','Page', ' Touch_Action ', ' X ', ' Y ', 'Norm_X', 'Norm_Y', 'Norm_DeltaX', 'Norm_DeltaY', 'Velocity', 'Acceleration',' Finger Size']]





