import matplotlib.pyplot as plt
import numpy as np

dat1 = np.load('mean_rewards_a2c_graph.npy')
dat2 = np.load('mean_rewards_graph.npy')

dat1_2 = []
for i in range(0,len(dat1)-3,3):
	dat1_2.append((dat1[i]+dat1[i+1]+dat1[i+2]+dat1[i+3])/4)

dat2_2 = []
for i in range(len(dat2)-3):
	dat2_2.append((dat2[i]+dat2[i+1]+dat2[i+2]+dat2[i+3])/4)
plt.plot(dat2_2)
plt.show()
