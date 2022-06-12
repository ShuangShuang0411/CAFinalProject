import matplotlib.pyplot as plt
import numpy as np


marker_size = 1;

data = np.loadtxt("./bin/data_evol.txt")

W0_orig = data[0]
W1_orig = data[1]
W2_orig = data[2]
W0_evol = data[3]
W1_evol = data[4]
W2_evol = data[5]

num = np.arange(len(W0_orig))

# s_speed = (5.0/3.0*data[5][370]/data[3][370])**0.5
# s_speed = (5.0/3.0*data[5][10]/data[3][10])**0.5
#print(s_speed,data[4][370],data[4][370]/s_speed)

evol_fig = plt.figure()            

plt.subplot(3,2,1)
plt.plot(num, W0_orig, 'o', color='red', ms=marker_size) 
plt.ylabel('density')
plt.title('Initial Condition')
plt.subplot(3,2,3)
plt.plot(num, W1_orig, 'o', color='green', ms=marker_size) 
plt.ylabel('velocity')
plt.subplot(3,2,5)
plt.plot(num, W2_orig, 'o', color='blue', ms=marker_size) 
plt.xlabel('position')
plt.ylabel('pressure')

plt.subplot(3,2,2)
plt.plot(num, W0_evol, 'o', color='red', ms=marker_size) 
plt.ylabel('density')
plt.title('Evolution')
plt.subplot(3,2,4)
plt.plot(num, W1_evol, 'o', color='green', ms=marker_size) 
plt.ylabel('velocity')
plt.subplot(3,2,6)
plt.plot(num, W2_evol, 'o', color='blue', ms=marker_size) 
plt.xlabel('position')
plt.ylabel('pressure')

plt.suptitle('Evolution of 1D Hydro acoustic wave')

plt.show()                   

