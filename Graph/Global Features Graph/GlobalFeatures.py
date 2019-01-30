import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\GlobalFeatures.csv', parse_dates=[2])

df.plot.bar(x = 'User', y = ['H_Average Stroke Time','V_Average Stroke Time'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average_Stroke_Time.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average Mid location','V_Average Mid location'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Mid location.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average Mid location','V_Average Mid location'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Mid location.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average Mean_of_velocity_firsthalf_stroke','V_Average Mean_of_velocity_firsthalf_stroke'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Mean_of_velocity_firsthalf_stroke.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average Mean_of_velocity_secondhalf_stroke','V_Average Mean_of_velocity_secondhalf_stroke'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Mean_of_velocity_secondhalf_stroke.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average Number of Data points','V_Average Number of Data points'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Number of Data points.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average Standard-Deviation','V_Average Standard-Deviation'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Standard-Deviation.png")
plt.close()
#df.plot.bar(x = 'User', y = ['H_Average Accelertation','V_Average Accelertation'], fontsize=6)
#plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Accelertation.png")


df.plot.bar(x = 'User', y = ['H_Average Width','V_Average Width'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Width.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average Height','V_Average Height'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Height.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average Max Slope','V_Average Max Slope'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Max Slope.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average Min Slope','V_Average Min Slope'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Min Slope.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average Start_X','V_Average Start_X'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Start_X.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average Start_Y','H_Average Start_Y'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Start_Y.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average End_X','V_Average End_X'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average End_X.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average End_Y','V_Average End_Y'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average End_y.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average Average_Delta_X','V_Average Average_Delta_X'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Average_Delta_X.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average Average_Delta_Y','V_Average Average_Delta_Y'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Average_Delta_Y.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average FingerSize_FingerDown','V_Average FingerSize_FingerDown'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average FingerSize_FingerDown.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average FingerSize_FingerUp','V_Average FingerSize_FingerUp'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average FingerSize_FingerUp.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average FingerSize','V_Average FingerSize'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average FingerSize.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average Velocity','V_Average Velocity'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Velocity.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average slope','V_Average slope'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average slope.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average Attack Angle','V_Average Attack Angle'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Attack Angle.png")
plt.close()

df.plot.bar(x = 'User', y = ['H_Average Leaving Angle','V_Average Leaving Angle'], fontsize=6)
plt.savefig("C:\\Users\\ee244\\Desktop\\Data Collection\\Analysis\\Features\\Average Leaving Angle.png")
plt.close()


