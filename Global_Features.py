import math
from datetime import datetime
import pandas as pd
import numpy as np
from matplotlib.pyplot import *
import numpy.linalg as la
import datetime as dt
import csv
from datetime import datetime
from scipy.interpolate import *
from numpy import *

class Global_Features:

    def Start_X(dataframe):
        Start_X_Value = dataframe.iat[0, 4]
        return Start_X_Value



#slicing rows
df = pd.read_csv('C:\\Users\\ee244\\Desktop\\Data Collection\\Data\\user128\\session 1\\features\\Features_S1.csv', parse_dates=[2])
df.columns=['Page','Subject_ID',' Touch_Action ',' Timestamp ',' X ',' Y ' ,'Pointer_ID','Tool_type','Orientation',' Pressure ',' Finger Size','Position',	'Delta_X',	'Delta_Y',	'Delta_Time',	'Slope',	'Velocity', 'Acceleration',	'Euc_Between_Points',	'Total_Stroke_Time',	'Peak_Velocity_Value',	'Mid-Location',	'Mean_of_velocity_firsthalf_stroke',	'Mean_of_velocity_secondhalf_stroke',	'Standard-Deviation',	'Average_Acceleration',	'Inking_Distance',	'Width',	'Height',	'Min_Slope',	'Max Slope',	'Start_X',	'End_X',	'Start_Y',	'End_Y',	'Average_Delta_X',	'Average_Delta_Y',	'Finger_Pressure_FingerDown',	'Finger_Pressure_FingerUp',	'Average_Finger_Pressure',	'Mid_Action_Pressure',	'Finger_Size_FingerDown',	'Finger_Size_FingerUp'	,'Average_Finger_Size',	'Stroke_Area_Outer',	'Average_Velocity',	'Attack_Angle',	'Leaving_Angle','Number_of_Datapoints'
]
#Total columns and rows
print(df.shape)
#finding null values
print(df.isnull().sum())

#findding NaN values
print(df.isna().sum())

#Dropping the row when any value in timestamp is missing
#df.dropna(subset=[' Timestamp '], how='any').shape

#Timestamp Conversion
#df[' Timestamp '] = pd.to_datetime(df[' Timestamp '], format=' %y/%m/%d %H:%M:%S:%f ')


#Separating Horizontal strokes
df_horizontal = df[(df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_1 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_2 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_3 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_4 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_5 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_6 ')  |
                   (df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_1 ') | (df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_2 ') | (df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_3 ') | (df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_4 ') | (df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_5 ')  |
                   (df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_1 ') | (df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_2 ') | (df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_3 ') | (df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_4 ')  ]



#Separating Vertical Strokes
df_Vertical = df[(df['Page'] == 'S1_S1_Vertical_Scroll_1 ') | (df['Page'] == 'S1_S1_Vertical_Scroll_2 ') | (df['Page'] == 'S1_S1_Vertical_Scroll_3 ') | (df['Page'] == 'S1_S1_Vertical_Scroll_4 ') | (df['Page'] == 'S1_S1_Vertical_Scroll_5 ') | (df['Page'] == 'S1_S1_Vertical_Scroll_6 ')
                 |(df['Page'] == 'S1_S2_Vertical_1 ') | (df['Page'] == 'S1_S2_Vertical_2 ') | (df['Page'] == 'S1_S2_Vertical_3 ') | (df['Page'] == 'S1_S2_Vertical_4 ') |
                  (df['Page'] == 'Scenario_3_VerticalSwipe_Activity_1 ') | (df['Page'] == 'Scenario_3_VerticalSwipe_Activity_2 ') | (df['Page'] == 'Scenario_3_VerticalSwipe_Activity_3 ') | (df['Page'] == 'Scenario_3_VerticalSwipe_Activity_4 ') ]

ActionDown_DF = df_horizontal.loc[df[' Touch_Action '] == " ACTION_DOWN "].index.values
ActionUp_DF = df_horizontal.loc[df[' Touch_Action '] == " ACTION_UP "].index.values


#Horizontal

Average_Stroke_Time = df_horizontal['Total_Stroke_Time'].mean()
Average_Peak_Velocity_Value = df_horizontal['Peak_Velocity_Value'].mean()
Average_Mid_location = df_horizontal['Mid-Location'].mean()
Average_Mean_of_velocity_firsthalf_stroke = df_horizontal['Mean_of_velocity_firsthalf_stroke'].mean()
Average_Mean_of_velocity_secondhalf_stroke = df_horizontal['Mean_of_velocity_secondhalf_stroke'].mean()
Average_Mean_of_velocity_firsthalf_stroke = df_horizontal['Mean_of_velocity_firsthalf_stroke'].mean()
Average_Standard_Deviation = df_horizontal['Standard-Deviation'].mean()
Average_Accelertation = df_horizontal['Average_Acceleration'].mean()
Average_Width = df_horizontal['Width'].mean()
Average_Height = df_horizontal['Height'].mean()
Average_Max_Slope = df_horizontal['Max Slope'].mean()
Average_Min_Slope = df_horizontal['Min_Slope'].mean()
Average_Start_X = df_horizontal['Start_X'].mean()
Average_Start_Y = df_horizontal['Start_Y'].mean()
Average_End_X = df_horizontal['End_X'].mean()
Average_End_Y = df_horizontal['End_Y'].mean()
Average_Delta_X = df_horizontal['Delta_X'].mean()
Average_Delta_Y = df_horizontal['Delta_Y'].mean()
Average_FingerSize_FingerDown = df_horizontal['Finger_Size_FingerDown'].mean()
Average_FingerSize_FingerUp = df_horizontal['Finger_Size_FingerUp'].mean()
Average_FingerSize = df_horizontal['Average_Finger_Size'].mean()
Average_Velocity = df_horizontal['Velocity'].mean()
Average_Attack_Angle = df_horizontal['Attack_Angle'].mean()
Average_Leaving_Angle = df_horizontal['Leaving_Angle'].mean()
Average_slope = df_horizontal['Slope'].mean()
Average_Number_of_datapoints = df_horizontal['Number_of_Datapoints'].mean()

#Vertical

Average_Stroke_Time_V = df_Vertical['Total_Stroke_Time'].mean()
Average_Peak_Velocity_Value_V = df_Vertical['Peak_Velocity_Value'].mean()
Average_Mid_location_V = df_Vertical['Mid-Location'].mean()
Average_Mean_of_velocity_firsthalf_stroke_V = df_Vertical['Mean_of_velocity_firsthalf_stroke'].mean()
Average_Mean_of_velocity_secondhalf_stroke_V = df_Vertical['Mean_of_velocity_secondhalf_stroke'].mean()
Average_Mean_of_velocity_firsthalf_stroke_V = df_Vertical['Mean_of_velocity_firsthalf_stroke'].mean()
Average_Standard_Deviation_V = df_Vertical['Standard-Deviation'].mean()
Average_Accelertation_V = df_Vertical['Average_Acceleration'].mean()
Average_Width_V = df_Vertical['Width'].mean()
Average_Height_V = df_Vertical['Height'].mean()
Average_Max_Slope_V = df_Vertical['Max Slope'].mean()
Average_Min_Slope_V = df_Vertical['Min_Slope'].mean()
Average_Start_X_V= df_Vertical['Start_X'].mean()
Average_Start_Y_V = df_Vertical['Start_Y'].mean()
Average_End_X_V = df_Vertical['End_X'].mean()
Average_End_Y_V = df_Vertical['End_Y'].mean()
Average_Delta_X_V = df_Vertical['Delta_X'].mean()
Average_Delta_Y_V = df_Vertical['Delta_Y'].mean()
Average_FingerSize_FingerDown_V = df_Vertical['Finger_Size_FingerDown'].mean()
Average_FingerSize_FingerUp_V = df_Vertical['Finger_Size_FingerUp'].mean()
Average_FingerSize_V = df_Vertical['Average_Finger_Size'].mean()
Average_Velocity_V = df_Vertical['Velocity'].mean()
Average_Attack_Angle_V = df_Vertical['Attack_Angle'].mean()
Average_Leaving_Angle_V = df_Vertical['Leaving_Angle'].mean()
Average_slope_V = df_Vertical['Slope'].mean()
Average_Number_of_datapoints_V = df_Vertical['Number_of_Datapoints'].mean()

#Writing onto file


df.at[0, 'H_Average Stroke Time'] = Average_Stroke_Time
df.at[0, 'H_Average Peak_Velocity_Value'] =Average_Peak_Velocity_Value
df.at[0, 'H_Average Mid location'] = Average_Mid_location
df.at[0, 'H_Average Mean_of_velocity_firsthalf_stroke'] = Average_Mean_of_velocity_firsthalf_stroke
df.at[0, 'H_Average Mean_of_velocity_secondhalf_stroke']= Average_Mean_of_velocity_secondhalf_stroke
df.at[0, 'H_Average Number of Data points']= Average_Number_of_datapoints
df.at[0, 'H_Average Standard-Deviation']= Average_Standard_Deviation
df.at[0, 'H_Average Accelertation']= Average_Accelertation
df.at[0, 'H_Average Width']= Average_Width
df.at[0, 'H_Average Height']= Average_Height
df.at[0, 'H_Average Max Slope']= Average_Max_Slope
df.at[0, 'H_Average Min Slope']= Average_Min_Slope
df.at[0, 'H_Average Start_X']= Average_Start_X
df.at[0, 'H_Average Start_Y']= Average_Start_Y
df.at[0, 'H_Average End_X']= Average_End_X
df.at[0, 'H_Average End_Y']= Average_End_Y
df.at[0, 'H_Average Average_Delta_X']= Average_Delta_X
df.at[0, 'H_Average Average_Delta_Y']= Average_Delta_Y
df.at[0, 'H_Average FingerSize_FingerDown']= Average_FingerSize_FingerDown
df.at[0, 'H_Average FingerSize_FingerUp']= Average_FingerSize_FingerUp
df.at[0, 'H_Average FingerSize']= Average_FingerSize
df.at[0, 'H_Average Velocity']= Average_Velocity
df.at[0, 'H_Average slope']= Average_slope
df.at[0, 'H_Average Attack Angle']= Average_Attack_Angle
df.at[0, 'H_Average Leaving Angle']= Average_Leaving_Angle



#Vertical

df.at[0, 'V_Average Stroke Time'] = Average_Stroke_Time_V
df.at[0, 'V_Average Peak_Velocity_Value'] =Average_Peak_Velocity_Value_V
df.at[0, 'V_Average Mid location'] = Average_Mid_location_V
df.at[0, 'V_Average Mean_of_velocity_firsthalf_stroke'] = Average_Mean_of_velocity_firsthalf_stroke_V
df.at[0, 'V_Average Mean_of_velocity_secondhalf_stroke']= Average_Mean_of_velocity_secondhalf_stroke_V
df.at[0, 'V_Average Number of Data points']= Average_Number_of_datapoints_V
df.at[0, 'V_Average Standard-Deviation']= Average_Standard_Deviation_V
df.at[0, 'V_Average Accelertation']= Average_Accelertation_V
df.at[0, 'V_Average Width']= Average_Width_V
df.at[0, 'V_Average Height']= Average_Height_V
df.at[0, 'V_Average Max Slope']= Average_Max_Slope_V
df.at[0, 'V_Average Min Slope']= Average_Min_Slope_V
df.at[0, 'V_Average Start_X']= Average_Start_X_V
df.at[0, 'V_Average Start_Y']= Average_Start_Y_V
df.at[0, 'V_Average End_X']= Average_End_X_V
df.at[0, 'V_Average End_Y']= Average_End_Y_V
df.at[0, 'V_Average Average_Delta_X']= Average_Delta_X_V
df.at[0, 'V_Average Average_Delta_Y']= Average_Delta_Y_V
df.at[0, 'V_Average FingerSize_FingerDown']= Average_FingerSize_FingerDown_V
df.at[0, 'V_Average FingerSize_FingerUp']= Average_FingerSize_FingerUp_V
df.at[0, 'V_Average FingerSize']= Average_FingerSize_V
df.at[0, 'V_Average Velocity']= Average_Velocity_V
df.at[0, 'V_Average slope']= Average_slope_V
df.at[0, 'V_Average Attack Angle']= Average_Attack_Angle_V
df.at[0, 'V_Average Leaving Angle']= Average_Leaving_Angle_V


#Writing to the file
df.to_csv("C:\\Users\\ee244\\Desktop\\Data Collection\\Data\\user128\\session 1\\features\\GlobalFeatures_S1.csv", index=False)


