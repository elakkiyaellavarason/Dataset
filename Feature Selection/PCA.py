

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jedi.refactoring import inline
from plotly.graph_objs import Bar, Scatter, Data, Layout, YAxis, Figure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math


#Loading Data from file
df = pd.read_csv('C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\PCA\\New_Features_PCA_input.csv', header=1,sep=',')
df.columns=['Total_Stroke_Time','Peak_Velocity_Value','Mid-Location','Mean_of_velocity_firsthalf_stroke',
            'Mean_of_velocity_secondhalf_stroke','Standard-Deviation','Average_Acceleration','Inking_Distance',
            'Width','Height','Min_Slope','Max Slope','Start_X','End_X','Start_Y','End_Y','Average_Delta_X',
            'Average_Delta_Y','Finger_Pressure_FingerDown',	'Finger_Pressure_FingerUp', 'Average_Finger_Pressure',
            'Mid_Action_Pressure','Finger_Size_FingerDown','Finger_Size_FingerUp',
            'Average_Finger_Size','Stroke_Area_Outer','Average_Velocity','Attack_Angle','Leaving_Angle','User']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

print(df.head())

features = ['Total_Stroke_Time',	'Peak_Velocity_Value',	'Mid-Location',	'Mean_of_velocity_firsthalf_stroke',	'Mean_of_velocity_secondhalf_stroke',	'Standard-Deviation','Average_Acceleration',	'Inking_Distance',	'Width','Height',	'Min_Slope',	'Max Slope','Start_X','End_X','Start_Y','End_Y','Average_Delta_X','Average_Delta_Y','Finger_Pressure_FingerDown','Finger_Pressure_FingerUp','Average_Finger_Pressure','Mid_Action_Pressure','Finger_Size_FingerDown',	'Finger_Size_FingerUp',	'Average_Finger_Size','Stroke_Area_Outer','Average_Velocity',	'Attack_Angle',	'Leaving_Angle']
x = df.loc[:, features].values
y = df.loc[:,['User']].values
x = StandardScaler().fit_transform(x)
print(pd.DataFrame(data = x, columns = features).head())
mean_vec = np.mean(x, axis=0)
cov_mat = (x - mean_vec).T.dot((x - mean_vec)) / (x.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
cov_mat = np.cov(x.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)


#Principal Components
pca = PCA(n_components=9)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3', 'principal component 4','principal component 5', 'principal component 6','principal component 7', 'principal component 8', 'principal component 9'])

print(principalDf.head(5))

print(df[['User']].head())

finalDf = pd.concat([principalDf, df[['User']]], axis = 1)
print('final Df:', finalDf.head(5))

#Visualing the importance of principal components
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
#eig_pairs.sort(reverse=True, key=(lambda x: x[0]))
#eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    plt.bar(range(29), var_exp, alpha=0.5, align='center',label='individual explained variance')
    plt.step(range(29), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


#Principal Components  Correlation with the original values
components = pd.DataFrame(pca.components_, columns = (df.columns[0],df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5],df.columns[6],df.columns[7],df.columns[8],
                                                      df.columns[9],df.columns[10],df.columns[11],df.columns[12],df.columns[13],df.columns[14],df.columns[15],df.columns[16],df.columns[17],
                                                      df.columns[18],df.columns[19],df.columns[20],df.columns[21],df.columns[22],df.columns[23],df.columns[24],df.columns[25],df.columns[26],df.columns[27],df.columns[28]), index=[1, 2, 3, 4, 5, 6, 7, 8, 9])
print('components:',components)

# Writing to the file
components.to_csv("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\PCA\\New_Features_PCA_input_Components.csv",index=False)
#projection onto new space
T=x

#Identifying important features
def get_important_features(transformed_features, components_, columns):
    """
    This function will return the most "important"
    features so we can determine which have the most
    effect on multi-dimensional scaling
    """
    num_columns = len(columns) -1

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    # Sort each column by it's length. These are your *original*
    # columns, not the principal components.
    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    print ("Features by importance:\n", important_features)

get_important_features(T, pca.components_, df.columns.values)


#Graph to show direction

def draw_vectors(transformed_features, components_, columns):
    """
    This funtion will project your *original* features
    onto your principal component feature-space, so that you can
    visualize how "important" each one was in the
    multi-dimensional scaling
    """

    num_columns = len(columns) - 1

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    ax = plt.axes()

    for i in range(num_columns):
    # Use an arrow to project each original feature as a
    # labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
        plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)

    return ax

ax = draw_vectors(T, pca.components_, df.columns.values)
T_df = pd.DataFrame(T)
T_df.columns = ['component1', 'component2','component3','component4','component5','component6','component7','component8','component9','component10',
                'component11','component12', 'component13', 'component14', 'component15', 'component16', 'component17', 'component18','component19','component20',
                'component21', 'component22', 'component23', 'component24', 'component25', 'component26', 'component27','component28', 'component29']

T_df['color'] = 'y'
T_df.loc[T_df['component1'] > 125, 'color'] = 'g'
T_df.loc[T_df['component2'] > 125, 'color'] = 'r'

plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.scatter(T_df['component1'], T_df['component2'], color=T_df['color'], alpha=0.5)
plt.show()




fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)


targets = ['user148', 'user149', 'user150']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['User'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

print('PCA:',pca.explained_variance_ratio_)


#Get the cumulative sum
#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
plt.show()