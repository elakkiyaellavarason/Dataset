
import matplotlib.pyplot as plt
import numpy as np

x=[2,3,4,5]
y=[0.283113309,0.230941571,0.224927,0.12]

plt.xticks(np.arange(2, 6, step=1))

plt.xlabel('K Value')
plt.ylabel('Average EER')
plt.ylim(0,0.5)
plt.title('Average EER with different K Values')
plt.plot(x,y,  color='black', marker='o')
plt.show()
plt.close()
