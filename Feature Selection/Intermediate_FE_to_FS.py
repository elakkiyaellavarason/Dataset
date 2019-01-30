import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jedi.refactoring import inline
from plotly.graph_objs import Bar, Scatter, Data, Layout, YAxis, Figure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math


#Loading Data from file
df = pd.read_csv('C:\\Users\\ee244\\Desktop\\Data Collection\\Data\\user123\\session 1\\features\\Features_S1_Vertical_swipes.csv',sep=',', usecols=['Total_Stroke_Time',	'Peak_Velocity_Value', 	'Mid-Location',	'Mean_of_velocity_firsthalf_stroke',	'Mean_of_velocity_secondhalf_stroke',
                                                                                                                                                     'Standard-Deviation',	'Average_Acceleration',	'Inking_Distance',	'Width',	'Height',
                                                                                                                                                     'Min_Slope','Max Slope',	'Start_X',	'End_X',	'Start_Y',	'End_Y',	'Average_Delta_X',	'Average_Delta_Y',	'Finger_Pressure_FingerDown',
                                                                                                                                                     'Finger_Pressure_FingerUp',	'Average_Finger_Pressure',	'Mid_Action_Pressure',	'Finger_Size_FingerDown','Finger_Size_FingerUp',
                                                                                                                                                     'Average_Finger_Size',	'Stroke_Area_Outer',	'Average_Velocity',	'Attack_Angle',	'Leaving_Angle'
 ])

df.dropna(how="all", inplace=True) # drops the empty line at file-end

df['user']='user123'
print(df)


df_horizontal = pd.read_csv('C:\\Users\\ee244\\Desktop\\Data Collection\\Data\\user123\\session 1\\features\\Features_S1_Horizontal_swipes.csv',sep=',', usecols=['Total_Stroke_Time',	'Peak_Velocity_Value', 	'Mid-Location',	'Mean_of_velocity_firsthalf_stroke','Mean_of_velocity_secondhalf_stroke',
                                                                                                                                                     'Standard-Deviation',	'Average_Acceleration',	'Inking_Distance',	'Width',	'Height',
                                                                                                                                                     'Min_Slope','Max Slope',	'Start_X',	'End_X',	'Start_Y',	'End_Y',	'Average_Delta_X',	'Average_Delta_Y',	'Finger_Pressure_FingerDown',
                                                                                                                                                     'Finger_Pressure_FingerUp',	'Average_Finger_Pressure',	'Mid_Action_Pressure',	'Finger_Size_FingerDown','Finger_Size_FingerUp',
                                                                                                                                                     'Average_Finger_Size',	'Stroke_Area_Outer',	'Average_Velocity',	'Attack_Angle',	'Leaving_Angle'
 ])
df_horizontal.dropna(how="all", inplace=True) # drops the empty line at file-end

df_horizontal['user']='user123'
print(df_horizontal)

# Writing to the file
#df.to_csv("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\PCA\\PCA_Batch Processing\\AllUsersPCAInput.csv",index=False)

with open('C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\PCA\\PCA_Batch Processing\\AllUsersPCAInput_Vertical.csv', 'a') as f:
    df.to_csv(f, header=False)

with open('C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\PCA\\PCA_Batch Processing\\AllUsersPCAInput_Horizontal.csv', 'a') as f:
    df_horizontal.to_csv(f, header=False)


