import matplotlib.pyplot as plt
import numpy as np

num = np.arange(1000)

data = np.loadtxt("./bin/PPM_data_evol.txt")
analytical = np.loadtxt("./bin/Sod_Shock_Tube")

W0_orig = data[0]
W1_orig = data[1]
W2_orig = data[2]
W0_evol = data[3]
W1_evol = data[4]
W2_evol = data[5]

s_speed = (5.0/3.0*data[5][370]/data[3][370])**0.5
#print(s_speed,data[4][370],data[4][370]/s_speed)

r = []
rho = []
v = []
pres = []
# the analytical solution for Sod shock tube
for i in range(2000):
    r.append(analytical[i][0]*1000)
    rho.append(analytical[i][1])
    v.append(analytical[i][2])
    pres.append(analytical[i][5])


evol_fig = plt.figure()            

plt.subplot(3,2,1)
plt.plot(num, W0_orig, 'o', color='red') 
plt.ylabel('density')
plt.title('Initial Condition')
plt.subplot(3,2,3)
plt.plot(num, W1_orig, 'o', color='green') 
plt.ylabel('velocity')
plt.subplot(3,2,5)
plt.plot(num, W2_orig, 'o', color='blue') 
plt.xlabel('position')
plt.ylabel('pressure')

plt.subplot(3,2,2)
plt.plot(num, W0_evol, 'o', color='red') 
plt.plot(r, rho, '-', color="black")
plt.ylabel('density')
plt.title('Evolution')
plt.subplot(3,2,4)
plt.plot(num, W1_evol, 'o', color='green') 
plt.plot(r, v, '-', color="black")
plt.ylabel('velocity')
plt.subplot(3,2,6)
plt.plot(num, W2_evol, 'o', color='blue') 
plt.plot(r, pres, '-', color="black")
plt.xlabel('position')
plt.ylabel('pressure')

plt.suptitle('Evolution of 1D Hydro tube')

plt.show()                   

