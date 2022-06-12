#--------------------------------------------------------------
# Plot the evolved data of Acoustic wave and order of accuracy
#--------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import math

num = np.linspace(0, 1, num=16)

data = np.loadtxt("./bin/data_evol.txt")

#Error = np.array([3.322219e-05,1.417730e-05,6.559202e-06,3.118763e-06,1.480458e-06,6.566155e-07])  # cha slope
#Error = np.array([5.002448e-05,2.597786e-05,1.391079e-05,8.172119e-06,4.464238e-06,2.211547e-06])  # cha whole
#Error = np.array([3.232024e-05,1.547775e-05,6.706204e-06,9.448957e-06,5.511531e-06,3.128637e-06])
Error = np.array([3.322222e-05,1.417714e-05,6.559126e-06,3.118726e-06,1.480440e-06,6.566069e-07])  # PPM
#Error = np.array([5.756340e-05,2.595956e-05,1.119514e-05,4.933086e-06,2.316238e-06,9.764499e-07])  # cha whole + orig. slope
#Error = np.array([3.322218e-05,1.417716e-05,6.559130e-06,3.118727e-06,1.480440e-06,6.566069e-07])  # PLM

Num = np.array([16,32,64,128,256,528])
dx = 1.0/Num
plt.plot(1.0/Num,Error, '--bo')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('dx')
plt.ylabel('error')
plt.show()

print ("slope of error of PPM method =", (math.log(Error[4])-math.log(Error[3]))/(math.log(dx[4])-math.log(dx[3])))
print ("slope of error of PPM method =", (math.log(Error[3])-math.log(Error[0]))/(math.log(dx[3])-math.log(dx[0])))
print ("slope of error of PPM method =", (math.log(Error[4])-math.log(Error[0]))/(math.log(dx[4])-math.log(dx[0])))

W0_orig = data[0]
W1_orig = data[1]
W2_orig = data[2]
W0_evol = data[3]
W1_evol = data[4]
W2_evol = data[5]


'''
evol_fig = plt.figure()            

plt.subplot(3,2,1)
plt.scatter(num, W0_orig, 3, color='red') 
plt.ylabel('density')
plt.title('Initial Condition')
plt.subplot(3,2,3)
plt.scatter(num, W1_orig, 3, color='green') 
plt.ylabel('velocity')
plt.subplot(3,2,5)
plt.scatter(num, W2_orig, 3, color='blue') 
plt.xlabel('position')
plt.ylabel('pressure')

plt.subplot(3,2,2)
plt.scatter(num, W0_evol, 3, color='red') 
#plt.plot(r, rho, '-', color="black")
plt.ylabel('density')
plt.title('Evolution')
plt.subplot(3,2,4)
plt.scatter(num, W1_evol, 3, color='green') 
#plt.plot(r, v, '-', color="black")
plt.ylabel('velocity')
plt.subplot(3,2,6)
plt.scatter(num, W2_evol, 3, color='blue') 
#plt.plot(r, pres, '-', color="black")
plt.xlabel('position')
plt.ylabel('pressure')

plt.suptitle('Evolution of 1D Acoustic Wave')

plt.show()                   
'''
