import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
from matplotlib.patches import FancyArrowPatch
import numpy as np
fig=plt.figure()

# slicing rows
df = pd.read_csv('C:\\Users\\ee244\\Desktop\\Data Collection\\Data\\user116\\session 1\\Touch.CSV', parse_dates=[2])
df.columns=['Page','Subject_ID','Touch_Action','Timestamp','X','Y' ,'Pointer_ID','Tool_type','Orientation','Pressure','Finger Size']

# Total columns and rows
#print(df.shape)
# finding null values
print(df.isnull().sum())

# findding NaN values
print(df.isna().sum())

# Dropping the row when any value in timestamp is missing
df.dropna(subset=['Timestamp'], how='any').shape

# Timestamp Conversion
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format=' %y/%m/%d %H:%M:%S:%f ')

#Separating Horizontal strokes
df_horizontal = df[(df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_1 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_2 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_3 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_4 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_5 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_6 ')  |
                   (df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_1 ') | (df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_2 ') | (df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_3 ') | (df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_4 ') | (df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_5 ')  |
                   (df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_1 ') | (df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_2 ') | (df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_3 ') | (df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_4 ')  ]

#Separating Horizontal stroke based on scenarios
df_h_s1_s1 = df[(df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_1 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_2 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_3 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_4 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_5 ') | (df['Page'] == 'S1_S1_Horizontal_Swipe_Activity_6 ')  ]
df_h_s1_s2 = df[(df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_1 ') | (df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_2 ') | (df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_3 ') | (df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_4 ') | (df['Page'] == 'S1_S2_Horizontal_Swipe_Activity_5 ')]
df_h_s1_s3 = df[(df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_1 ') | (df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_2 ') | (df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_3 ') | (df['Page'] == 'S1_S3_Horizontal_Swipe_Activity_4 ')]

print('DF S1 S1', len(df_h_s1_s1))
#Separating Vertical Strokes
df_Vertical = df[(df['Page'] == 'S1_S1_Vertical_Scroll_1 ') | (df['Page'] == 'S1_S1_Vertical_Scroll_2 ') | (df['Page'] == 'S1_S1_Vertical_Scroll_3 ') | (df['Page'] == 'S1_S1_Vertical_Scroll_4 ') | (df['Page'] == 'S1_S1_Vertical_Scroll_5 ') | (df['Page'] == 'S1_S1_Vertical_Scroll_6 ')
                 |(df['Page'] == 'S1_S2_Vertical_1 ') | (df['Page'] == 'S1_S2_Vertical_2 ') | (df['Page'] == 'S1_S2_Vertical_3 ') | (df['Page'] == 'S1_S2_Vertical_4 ') |
                  (df['Page'] == 'Scenario_3_VerticalSwipe_Activity_1 ') | (df['Page'] == 'Scenario_3_VerticalSwipe_Activity_2 ') | (df['Page'] == 'Scenario_3_VerticalSwipe_Activity_3 ') | (df['Page'] == 'Scenario_3_VerticalSwipe_Activity_4 ') ]
#Separating vertical strokes based on scenarios
df_v_s1_s1 = df[(df['Page'] == 'S1_S1_Vertical_Scroll_1 ') | (df['Page'] == 'S1_S1_Vertical_Scroll_2 ') | (df['Page'] == 'S1_S1_Vertical_Scroll_3 ') | (df['Page'] == 'S1_S1_Vertical_Scroll_4 ') | (df['Page'] == 'S1_S1_Vertical_Scroll_5 ') | (df['Page'] == 'S1_S1_Vertical_Scroll_6 ')]
df_v_s1_s2 = df[(df['Page'] == 'S1_S2_Vertical_1 ') | (df['Page'] == 'S1_S2_Vertical_2 ') | (df['Page'] == 'S1_S2_Vertical_3 ') | (df['Page'] == 'S1_S2_Vertical_4 ')]
df_v_s1_s3 = df[(df['Page'] == 'Scenario_3_VerticalSwipe_Activity_1 ') | (df['Page'] == 'Scenario_3_VerticalSwipe_Activity_2 ') | (df['Page'] == 'Scenario_3_VerticalSwipe_Activity_3 ') | (df['Page'] == 'Scenario_3_VerticalSwipe_Activity_4 ') ]



'''
#Horizontal Swipes session 2
df_horizontal = df[(df['Page '] == "S2_S1_Horizontal_Swipe_Activity_1 " ) | (df['Page '] == "S2_S1_Horizontal_Swipe_Activity_2 ") |  (df['Page '] == "S2_S1_Horizontal_Swipe_Activity_3 ") |  (df['Page '] == "S2_S1_Horizontal_Swipe_Activity_4 ") |  (df['Page '] == "S2_S1_Horizontal_Swipe_Activity_5 ") |
                    (df['Page '] == "S2_S2_Horizontal_Swipe_Activity_1 ") | (df['Page '] == "S2_S2_Horizontal_Swipe_Activity_2 ") |  (df['Page '] == "S2_S3_Horizontal_Swipe_Activity_3 ") |  (df['Page '] == "S2_S4_Horizontal_Swipe_Activity_4 ") |  (df['Page '] == "S2_S5_Horizontal_Swipe_Activity_5 ") |
                   (df['Page '] == "S2_S3_Horizontal_Swipe_Activity_1 ") | (
                               df['Page '] == "S2_S3_Horizontal_Swipe_Activity_2 ") |
                               (df['Page '] == "S2_S3_Horizontal_Swipe_Activity_3 ") | (
                              df['Page '] == "S2_S3_Horizontal_Swipe_Activity_4 ")]
'''
#print(df_horizontal)
#Horizontal Swipes
#df_horizontal = df.groupby(['Page ']).get_group('S1_S1_Horizontal_Swipe_Activity_1 ')

ActionDown_DF_S1 = df_h_s1_s1.loc[df['Touch_Action'] == " ACTION_DOWN "].index.values
ActionUp_DF_S1 = df_h_s1_s1.loc[df['Touch_Action'] == " ACTION_UP "].index.values

ActionDown_DF_S2 = df_h_s1_s2.loc[df['Touch_Action'] == " ACTION_DOWN "].index.values
ActionUp_DF_S2 = df_h_s1_s2.loc[df['Touch_Action'] == " ACTION_UP "].index.values

ActionDown_DF_S3 = df_h_s1_s3.loc[df['Touch_Action'] == " ACTION_DOWN "].index.values
ActionUp_DF_S3 = df_h_s1_s3.loc[df['Touch_Action'] == " ACTION_UP "].index.values

ActionDown_DF_V_S1 = df_v_s1_s1.loc[df['Touch_Action'] == " ACTION_DOWN "].index.values
ActionUp_DF_V_S1 = df_v_s1_s1.loc[df['Touch_Action'] == " ACTION_UP "].index.values

ActionDown_DF_V_S2 = df_v_s1_s2.loc[df['Touch_Action'] == " ACTION_DOWN "].index.values
ActionUp_DF_V_S2 = df_v_s1_s2.loc[df['Touch_Action'] == " ACTION_UP "].index.values

ActionDown_DF_V_S3 = df_v_s1_s3.loc[df['Touch_Action'] == " ACTION_DOWN "].index.values
ActionUp_DF_V_S3 = df_v_s1_s3.loc[df['Touch_Action'] == " ACTION_UP "].index.values

#print("Length of Action_UP", len(ActionUp_DF))
#print("Length of Action_Down", len(ActionDown_DF))
plt.xlabel('Time')
plt.ylabel('Finger Size')
plt.title(' Swipe Strokes ')
plt.xlim(0,1700)
plt.ylim(0,1700)


#fig = plt.figure()
#ax = fig.add_subplot(111)

#fig = plt.figure()
#ax = fig.add_subplot(111)
fig = plt.figure()
ax = fig.add_subplot(111)
####################################################S1_S1#################################################
#@@@@@@@@@@@@@@@@@@@Horizontal@@@@@@@@@@@@@@@@@@@
count = 0
while (count <= 20):

    if (count != len(ActionUp_DF_S2)):
        #print('The count is:', count)
        df_segment = df.iloc[ActionDown_DF_S2[count]:ActionUp_DF_S2[count] + 1, 0:7]
       # print(df_segment)
        x_values = df_segment['X']
        y_values = df_segment['Y']

        plt.plot(x_values, y_values, label=count)

        # get x and y vectors
        x = np.array(x_values)
        y = np.array(y_values)

        # calculate polynomial
        z = np.polyfit(x, y, 3)
        f = np.poly1d(z)

        # calculate new x's and y's
        x_new = np.linspace(x[0], x[-1], 50)
        y_new = f(x_new)

        plt.plot(x, y, 'o', x_new, y_new)

        plt.plot(x, y)

        #ar = FancyArrowPatch((x_values[count - 1], y_values[count - 1]), (x_values[count], y_values[count]),
        #                     arrowstyle='->', mutation_scale=20)
       # ax.add_patch(ar)

        #for i, j in zip(x_values, y_values):
          # ax.annotate('%s)' % j, xy=(i, j), xytext=(30, 0), textcoords='offset points')
          # ax.annotate('(%s,' % i, xy=(i, j))

    count = count + 1
plt.show()

fig.savefig('C:\\Users\\ee244\\Desktop\\Data Collection\\Data\\user116\\session 1\\graphs\\polyfit\\Horizontal_swipes_S1_s2'+'.png')

'''
#@@@@@@@@@@@@@@@@@@@Vertical@@@@@@@@@@@@@@@@@@@
fig = plt.figure()
ax = fig.add_subplot(111)
count = 0
while (count <= len(ActionDown_DF_V_S1)):

    if (count != len(ActionDown_DF_V_S1)):
        #print('The count is:', count)
        df_segment = df.iloc[ActionDown_DF_V_S1[count]:ActionUp_DF_V_S1[count] + 1, 0:7]
       # print(df_segment)
        x_values = df_segment['X']
        y_values = df_segment['Y']

        plt.plot(x_values, y_values, label=count)


        for i, j in zip(x_values, y_values):
          ax.annotate('%s)' % j, xy=(i, j), xytext=(30, 0), textcoords='offset points')
          ax.annotate('(%s,' % i, xy=(i, j))

    count = count + 1
plt.show()

#fig.savefig('C:\\Users\\ee244\\Desktop\\Data Collection\\Data\\user116\\session 1\\graphs\\Vertical_swipes_S1_s1'+'.png')

####################################################S1_S2#################################################

fig = plt.figure()
ax = fig.add_subplot(111)
count = 0
while (count <= len(ActionUp_DF_S2)):

    if (count != len(ActionUp_DF_S2)):
        #print('The count is:', count)
        df_segment = df.iloc[ActionDown_DF_S2[count]:ActionUp_DF_S2[count] + 1, 0:7]
       # print(df_segment)
        x_values = df_segment['X']
        y_values = df_segment['Y']

        plt.plot(x_values, y_values, label=count)


        #for i, j in zip(x_values, y_values):
         #   ax.annotate('%s)' % j, xy=(i, j), xytext=(30, 0), textcoords='offset points')
         #   ax.annotate('(%s,' % i, xy=(i, j))

    count = count + 1
plt.show()

#fig.savefig('C:\\Users\\ee244\\Desktop\\Data Collection\\Data\\user125\\session 1\\graphs\\Horizontal_swipes_S1_s2'+'.png')

#@@@@@@@@@@@@@@@@@@@Vertical@@@@@@@@@@@@@@@@@@@
fig = plt.figure()
ax = fig.add_subplot(111)
count = 0
while (count <= len(ActionDown_DF_V_S2)):

    if (count != len(ActionDown_DF_V_S2)):
        #print('The count is:', count)
        df_segment = df.iloc[ActionDown_DF_V_S2[count]:ActionUp_DF_V_S2[count] + 1, 0:7]
       # print(df_segment)
        x_values = df_segment['X']
        y_values = df_segment['Y']

        plt.plot(x_values, y_values, label=count)


        #for i, j in zip(x_values, y_values):
          # ax.annotate('%s)' % j, xy=(i, j), xytext=(30, 0), textcoords='offset points')
          # ax.annotate('(%s,' % i, xy=(i, j))

    count = count + 1
plt.show()

#fig.savefig('C:\\Users\\ee244\\Desktop\\Data Collection\\Data\\user125\\session 1\\graphs\\Vertical_swipes_s1_s2'+'.png')
'''
'''
####################################################S1_S3#################################################
fig = plt.figure()
ax = fig.add_subplot(111)
count = 0
while (count <= len(ActionUp_DF_S3)):

    if (count != len(ActionUp_DF_S3)):
        #print('The count is:', count)
        df_segment = df.iloc[ActionDown_DF_S3[count]:ActionUp_DF_S3[count] + 1, 0:7]
       # print(df_segment)
        x_values = df_segment['X']
        y_values = df_segment['Y']

        plt.plot(x_values, y_values, label=count)


        for i, j in zip(x_values, y_values):
           ax.annotate('%s)' % j, xy=(i, j), xytext=(30, 0), textcoords='offset points')
           ax.annotate('(%s,' % i, xy=(i, j))

    count = count + 1
plt.show()
'''

''' 
labels = ['text{}'.format(i) for i in range(len(x_values))]
for label, x, y in zip(labels, x_values, y_values):
    plt.annotate(label, xy=(x, y), xytext=(x_values, y_values))
plt.show()
'''

fig.savefig('C:\\Users\\ee244\\Desktop\\Data Collection\\Data\\user103\\session 1\\graphs\\Horizontal_swipes_s1_s3_trial'+'.png')
'''
#@@@@@@@@@@@@@@@@@@@Vertical@@@@@@@@@@@@@@@@@@@
fig = plt.figure()
ax = fig.add_subplot(111)
count = 0
while (count <= len(ActionDown_DF_V_S3)):

    if (count != len(ActionDown_DF_V_S3)):
        #print('The count is:', count)
        df_segment = df.iloc[ActionDown_DF_V_S3[count]:ActionUp_DF_V_S3[count] + 1, 0:7]
       # print(df_segment)
        x_values = df_segment['X']
        y_values = df_segment['Y']

        plt.plot(x_values, y_values, label=count)


        #for i, j in zip(x_values, y_values):
          #ax.annotate('%s)' % j, xy=(i, j), xytext=(30, 0), textcoords='offset points')
         # ax.annotate('(%s,' % i, xy=(i, j))

    count = count + 1
plt.show()
'''
#fig.savefig('C:\\Users\\ee244\\Desktop\\Data Collection\\Data\\user125\\session 1\\graphs\\Vertical_swipes_s1_s3'+'.png')
