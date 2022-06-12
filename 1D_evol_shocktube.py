#-----------------------------------------------------------
# Plot the evolved data of shock tube problem
#-----------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

num = np.linspace(0, 1, num=1000)

data = np.loadtxt("./bin/data_evol.txt")
analytical = np.loadtxt("./Sod_Shock_Tube")

W0_orig = data[0]
W1_orig = data[1]
W2_orig = data[2]
W0_evol = data[3]
W1_evol = data[4]
W2_evol = data[5]

r = []
rho = []
v = []
pres = []
# the analytical solution for Sod shock tube
for i in range(2000):
    r.append(analytical[i][0])
    rho.append(analytical[i][1])
    v.append(analytical[i][2])
    pres.append(analytical[i][5])

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
plt.plot(r, rho, '-', color="black")
plt.ylabel('density')
plt.title('Evolution')
plt.subplot(3,2,4)
plt.scatter(num, W1_evol, 3, color='green') 
plt.plot(r, v, '-', color="black")
plt.ylabel('velocity')
plt.subplot(3,2,6)
plt.scatter(num, W2_evol, 3, color='blue') 
plt.plot(r, pres, '-', color="black")
plt.xlabel('position')
plt.ylabel('pressure')

plt.suptitle('Evolution of 1D Hydro tube')

plt.show()                   

