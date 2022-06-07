import matplotlib.pyplot as plt
import numpy as np
import math

N = np.array([50, 100, 200, 500, 1000, 2000])

data = np.loadtxt("./bin/ac_data_evol.txt")
analytical = np.loadtxt("./bin/Sod_Shock_Tube")

rho1 = data[0]
rho2 = data[3]
rho3 = data[6]
rho4 = data[9]
rho5 = data[12]
rho6 = data[15]

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

err1 = 0.0
for i in range(50):
    err1 += abs(rho[i*40] - rho1[i]) 
err1 /= 50.0
print(err1)

err2 = 0.0 
for i in range(100):
    err2 += abs(rho[i*20] - rho2[i]) 
err2 /= 100.0
print(err2)

err3 = 0.0 
for i in range(200):
    err3 += abs(rho[i*10] - rho3[i]) 
err3 /= 200.0
print(err3)

err4 = 0.0 
for i in range(500):
    err4 += abs(rho[i*4] - rho4[i]) 
err4 /= 500.0
print(err4)

err5 = 0.0 
for i in range(1000):
    err5 += abs(rho[i*2] - rho5[i]) 
err5 /= 1000.0
print(err5)

err6 = 0.0 
for i in range(2000):
    err6 += abs(rho[i] - rho6[i]) 
err6 /= 2000.0
print(err6)


Error = np.array([err1, err2, err3, err4, err5, err6])
dx = 1.0/N
print ("slope of error of PPM method =", (math.log(Error[5])-math.log(Error[3]))/(math.log(dx[5])-math.log(dx[3])))
print ("slope of error of PPM method =", (math.log(Error[3])-math.log(Error[0]))/(math.log(dx[3])-math.log(dx[0])))
print ("slope of error of PPM method =", (math.log(Error[5])-math.log(Error[0]))/(math.log(dx[5])-math.log(dx[0])))

plt.plot(dx, Error, '--bo')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('dx')
plt.ylabel('error')
plt.show()

